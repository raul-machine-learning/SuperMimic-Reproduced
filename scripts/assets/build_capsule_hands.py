"""Two floating capsule hands built from the sweep mocap skeleton, and the mocap replayed on them.

  uv run --with h5py python scripts/assets/build_capsule_hands.py

The hands are the actor's own (real scale, no robot in between). Each hand is a tree of the mocap's hand bones,
rooted at the ``{Left,Right}Hand`` bone on a free joint:

  palm      Hand + the four InHand* metacarpal bones (they rotate < 1 deg in the recording: rigid)
  fingers   *1 knuckle: 2 hinges (flexion, spread), *2 and *3: 1 hinge each
  thumb     Thumb1: 2 hinges, Thumb2: 2 hinges, Thumb3: 1 hinge

Hinge axes come from the recorded local bone rotations: the principal axes of their rotation vectors, flexion
first (signed so the clip's mean flexion is positive). The angles are the intrinsic Euler decomposition of each
local rotation about those axes; the dropped third axis is under 1 deg in this recording. Every bone is a capsule
from its joint to its child's; the palm is capsules from the wrist to each metacarpal and along the metacarpals.
Fingers are position-driven (effort-limited, like tendons); the free roots are driven by whoever uses the model.

Writes ``src/supermimic/assets/capsule_hands/capsule_hands.xml`` and ``data/hands/<clip>.npz`` holding, on the
mocap's time grid: ``base_pos`` / ``base_quat`` (T, 2, 3/4) wrist poses (left, right), ``joint_pos`` (T, nj)
in the model's hinge order, the props, the table, the glove pressure and the mocap hand-bone positions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import tyro
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
RETARGET_ROOT = Path.home() / "g1-sweep-retarget"
ASSET = ROOT / "src/supermimic/assets/capsule_hands/capsule_hands.xml"

SIDES = {"L": "Left", "R": "Right"}
FINGERS = ("Index", "Middle", "Ring", "Pinky")
# bone suffix -> (number of hinges, upper flexion limit in rad: room to close beyond the recording)
FINGER_JOINTS = {"1": (2, 1.60), "2": (1, 1.90), "3": (1, 1.30)}
THUMB_JOINTS = {"1": (2, None), "2": (2, None), "3": (1, 1.30)}
# capsule radii (m) of the phalanges, proximal to distal
FINGER_RADIUS = {"Index": (0.0090, 0.0082, 0.0074), "Middle": (0.0093, 0.0085, 0.0076),
                 "Ring": (0.0088, 0.0080, 0.0072), "Pinky": (0.0078, 0.0071, 0.0065)}
THUMB_RADIUS = (0.0120, 0.0105, 0.0095)
METACARPAL_RADIUS = {"Index": 0.0110, "Middle": 0.0110, "Ring": 0.0105, "Pinky": 0.0095}
WRIST_RADIUS = 0.0120
THENAR_RADIUS = 0.0130
# per hinge kind: effort limit (N m), stiffness (N m / rad); damping gives a damping ratio ~1 on the armature
EFFORT = {"flex1": 2.0, "spread1": 1.0, "flex2": 1.0, "flex3": 0.5, "thumb_flex1": 2.0, "thumb_spread1": 2.0,
          "thumb_flex2": 1.5, "thumb_spread2": 1.0, "thumb_flex3": 0.8}
ARMATURE = 2.0e-3
SKIN = [0.87, 0.72, 0.62, 1.0]
SKIN_R = [0.80, 0.66, 0.58, 1.0]


def local_frames(human, b: int) -> tuple[Rotation, np.ndarray]:
    """Local rotation (T) and position (T, 3) of bone b in its parent's frame."""
    p = human.parents[b]
    rp = Rotation.from_quat(human.quat[:, p], scalar_first=True)
    return rp.inv() * Rotation.from_quat(human.quat[:, b], scalar_first=True), rp.inv().apply(human.pos[:, b] - human.pos[:, p])


def hinge_axes(r: Rotation) -> np.ndarray:
    """(3, 3) rows: principal axes of the rotation vectors (uncentered), flexion first, mean flexion >= 0."""
    rv = r.as_rotvec()
    w, v = np.linalg.eigh(rv.T @ rv / len(rv))
    a = v[:, ::-1].T.copy()
    if (rv @ a[0]).mean() < 0:
        a[0] = -a[0]
    a[2] = np.cross(a[1], a[0])
    return a


def decompose(r: Rotation, axes: np.ndarray) -> tuple[np.ndarray, float]:
    """Angles (T, 3) with r = R(a0, t0) R(a1, t1) R(a2, t2) (intrinsic), and the max |t2| in degrees."""
    C = np.stack([axes[2], axes[1], axes[0]], 1)  # columns x', y', z' = a2, a1, a0
    Rc = Rotation.from_matrix(C)
    e = (Rc.inv() * r * Rc).as_euler("ZYX")  # intrinsic z' (a0), y' (a1), x' (a2)
    return e, float(np.degrees(np.abs(e[:, 2]).max()))


def add_capsule(body, name: str, a: np.ndarray, b: np.ndarray, radius: float, rgba, shorten_end: float = 0.0) -> None:
    d = b - a
    L = np.linalg.norm(d)
    b = a + d * max(L - shorten_end, 1e-3) / L
    body.add_geom(name=f"{name}_collision", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[*a, *b], size=[radius, 0, 0],
                  rgba=rgba, density=1000.0, condim=4, friction=[1.0, 0.005, 0.0001])


def main(mocap_dir: Path = ROOT / "data/sweep_mocap", retarget_root: Path = RETARGET_ROOT, clip: str = "sweep_task023_hands",
         out_dir: Path = ROOT / "data/hands", asset: Path = ASSET) -> None:
    sys.path.insert(0, str(retarget_root))
    from sweep_retarget.mocap import load_human, load_pressure, load_props, raw_timestamp_origin
    from sweep_retarget.scene import estimate_table

    human = load_human(mocap_dir)
    props = load_props(mocap_dir)
    pressure = load_pressure(mocap_dir, raw_timestamp_origin(mocap_dir))
    T = len(human.times)
    idx = human.index

    spec = mujoco.MjSpec()
    spec.modelname = "capsule_hands"
    spec.compiler.degree = False
    joint_angles: dict[str, np.ndarray] = {}
    report = []

    def child_pos(name: str) -> np.ndarray:
        return local_frames(human, idx(name))[1].mean(0)

    for S, side in SIDES.items():
        rgba = SKIN if S == "L" else SKIN_R
        hand = spec.worldbody.add_body(name=f"{S}_hand")
        hand.add_freejoint(name=f"{S}_hand_root")
        hand.add_site(name=f"{S}_wrist", size=[0.004, 0, 0])
        zero = np.zeros(3)
        add_capsule(hand, f"{S}_thenar", zero, child_pos(f"{side}HandThumb1"), THENAR_RADIUS, rgba)

        def chain(parent_body, bones: list[str], joints: dict, radii, prefix: str, tip: str) -> None:
            body_parent = parent_body
            for k, bone in enumerate(bones):
                r, lp = local_frames(human, idx(bone))
                b = body_parent.add_body(name=f"{prefix}{k + 1}", pos=lp.mean(0))
                n_h, flex_max = joints[str(k + 1)]
                axes = hinge_axes(r)
                ang, dropped = decompose(r, axes)
                kinds = ["flex", "spread"][:n_h]
                if n_h == 1:
                    dropped = max(dropped, float(np.degrees(np.abs(ang[:, 1]).max())))
                report.append((f"{prefix}{k + 1}", n_h, dropped))
                for h, kind in enumerate(kinds):
                    jn = f"{prefix}{k + 1}_{kind}"
                    lo, hi = ang[:, h].min(), ang[:, h].max()
                    if kind == "flex":
                        rng = [lo - 0.15, max(hi + 0.3, flex_max or hi + 0.5)]
                    else:
                        rng = [lo - 0.2, hi + 0.2]
                    b.add_joint(name=jn, type=mujoco.mjtJoint.mjJNT_HINGE, axis=axes[h], range=rng, armature=ARMATURE)
                    joint_angles[jn] = ang[:, h]
                nxt = bones[k + 1] if k + 1 < len(bones) else tip
                end = child_pos(nxt)
                add_capsule(b, f"{prefix}{k + 1}", zero, end, radii[k], rgba, shorten_end=radii[k] if nxt == tip else 0.0)
                if nxt == tip:
                    b.add_site(name=f"{prefix}_tip", pos=end, size=[0.004, 0, 0])
                body_parent = b

        for f in FINGERS:
            fl = f.lower()
            r0, lp0 = local_frames(human, idx(f"{side}InHand{f}"))
            mc = hand.add_body(name=f"{S}_{fl}0", pos=lp0.mean(0), quat=Rotation.from_rotvec(r0.as_rotvec().mean(0)).as_quat(scalar_first=True))
            add_capsule(hand, f"{S}_{fl}_wrist", zero, lp0.mean(0), WRIST_RADIUS, rgba)
            add_capsule(mc, f"{S}_{fl}0", zero, child_pos(f"{side}Hand{f}1"), METACARPAL_RADIUS[f], rgba)
            chain(mc, [f"{side}Hand{f}{k}" for k in (1, 2, 3)], FINGER_JOINTS, FINGER_RADIUS[f], f"{S}_{fl}", f"{side}Hand{f}3_End")
        chain(hand, [f"{side}HandThumb{k}" for k in (1, 2, 3)], THUMB_JOINTS, THUMB_RADIUS, f"{S}_thumb", f"{side}HandThumb3_End")

    # Phalanges and palm pieces of one hand never collide with each other (their capsules meet at the knuckles).
    for S in SIDES:
        bodies = [b.name for b in spec.bodies if b.name.startswith(f"{S}_")]
        for i, b1 in enumerate(bodies):
            for b2 in bodies[i + 1 :]:
                spec.add_exclude(bodyname1=b1, bodyname2=b2)

    for j in spec.joints:
        if j.type != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        kind = j.name.split("_", 1)[1]
        kind = ("thumb_" if kind.startswith("thumb") else "") + kind.split("_")[-1] + kind.split("_")[0][-1]
        effort = EFFORT[kind]
        kp = effort / 0.25
        kd = 2.0 * np.sqrt(kp * ARMATURE)
        a = spec.add_actuator(name=j.name, target=j.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.gainprm = [kp] + [0.0] * 9
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.biasprm = [0.0, -kp, -kd] + [0.0] * 7
        a.forcelimited, a.forcerange = True, [-effort, effort]
        a.ctrllimited, a.ctrlrange = True, j.range

    model = spec.compile()
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_text(spec.to_xml())
    hinges = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]

    base_pos = np.stack([human.p(f"{side}Hand") for side in SIDES.values()], 1)
    base_quat = np.stack([human.quat[:, idx(f"{side}Hand")] for side in SIDES.values()], 1)
    joint_pos = np.stack([joint_angles[n] for n in hinges], 1)

    # Replaying the angles must put every fingertip where the mocap has it.
    data = mujoco.MjData(model)
    tips = [(f"{S}_{f.lower()}_tip", f"{side}Hand{f}3_End") for S, side in SIDES.items() for f in FINGERS] + \
           [(f"{S}_thumb_tip", f"{side}HandThumb3_End") for S, side in SIDES.items()]
    err = np.zeros((T, len(tips)))
    qadr = [model.jnt_qposadr[model.joint(n).id] for n in hinges]
    for t in range(T):
        for k, S in enumerate(SIDES):
            a = model.jnt_qposadr[model.joint(f"{S}_hand_root").id]
            data.qpos[a : a + 3], data.qpos[a + 3 : a + 7] = base_pos[t, k], base_quat[t, k]
        data.qpos[qadr] = joint_pos[t]
        mujoco.mj_kinematics(model, data)
        for i, (site, bone) in enumerate(tips):
            err[t, i] = np.linalg.norm(data.site_xpos[model.site(site).id] - human.pos[t, idx(bone)])

    table = estimate_table(props.pos, human.p("Hips"))
    hand_bones = [n for n in human.names if "Hand" in n]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{clip}.npz"
    np.savez_compressed(
        out, times=human.times, base_pos=base_pos, base_quat=base_quat, joint_names=np.array(hinges), joint_pos=joint_pos,
        prop_names=np.array(list(props.pos)), prop_pos=np.stack([props.pos[n] for n in props.pos], 1),
        prop_quat=np.stack([props.quat[n] for n in props.quat], 1), table_center=table[0], table_half=table[1],
        pressure_times=pressure.times if pressure else np.zeros(0),
        pressure_left=pressure.left if pressure else np.zeros((0, 129), np.uint8),
        pressure_right=pressure.right if pressure else np.zeros((0, 129), np.uint8),
        human_hand_names=np.array(hand_bones), human_hand_pos=human.pos[:, [idx(n) for n in hand_bones]], mocap_dir=str(mocap_dir),
    )
    mass = {S: sum(model.body_mass[b] for b in range(model.nbody) if model.body(b).name.startswith(f"{S}_")) for S in SIDES}
    print(f"wrote {asset}: {len(hinges)} finger hinges, {model.nu} actuators, hand mass L {mass['L']:.3f} R {mass['R']:.3f} kg")
    print(f"wrote {out}: {T} frames ({human.times[-1]:.1f} s)")
    print(f"  fingertip replay error: mean {err.mean() * 1000:.1f} mm, max {err.max() * 1000:.1f} mm")
    worst = max(report, key=lambda x: x[2])
    print(f"  largest rotation left out by the hinges: {worst[2]:.1f} deg ({worst[0]})")
    for n in hinges:
        if n.startswith("L_"):
            j = model.joint(n)
            a = joint_angles[n]
            print(f"  {n:18s} data {np.degrees(a.min()):+6.1f}..{np.degrees(a.max()):+6.1f} deg   range {np.degrees(j.range[0]):+6.1f}..{np.degrees(j.range[1]):+6.1f}")


if __name__ == "__main__":
    tyro.cli(main)

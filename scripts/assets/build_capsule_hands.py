"""Two floating capsule hands built from the sweep mocap skeleton, and the mocap replayed on them.

  uv run python scripts/assets/build_capsule_hands.py

The hands are the actor's own, read straight from the mocap at real scale: no robot and no retargeting in between.
Each hand is a tree of the mocap's hand bones, rooted at the ``{Left,Right}Hand`` bone on a free joint:

  palm      Hand + the four InHand* metacarpal bones (they rotate < 1 deg in the recording: rigid)
  fingers   *1 knuckle: 2 hinges (flexion, spread), *2 and *3: 1 hinge each
  thumb     Thumb1: 2 hinges, Thumb2: 2 hinges, Thumb3: 1 hinge

Hinge axes come from the recorded local bone rotations: the principal axes of their rotation vectors, flexion
first. The angles are the intrinsic Euler decomposition of each local rotation about those axes; the dropped third
axis is under 1 deg in this recording. Flexion is signed geometrically: a positive angle moves the fingertip toward
the wrist (the thumb tip toward the middle finger's knuckle), at the clip's median pose. Every bone is a capsule from
its joint to its child's; the palm is one ray of two capsules per finger, from a heel point to the knuckle (the heel
points spread wider than the mocap's metacarpal bases, as the flesh is), a capsule across the heel, and the thenar.
With ``SM_HAND_SHAPE=box`` every capsule is a box instead: the same span (the radius past each end), a square
cross-section as wide as the capsule, one face out of the palm (thumb: toward where its own flexion moves it), at
the model's zero pose; the distal thumb box instead has a fixed roll angle about its bone, set anatomically at the
clip's median pose (``thumb_frames``; fitted by scripts/data/fit_glove_blocks.py).
Fingers are position-driven (effort-limited, like tendons); the free roots are driven by whoever uses the model.

Writes ``src/supermimic/assets/<shape>_hands/<shape>_hands.xml`` and ``<shape>_rig.xml``, and
``data/hands/<clip>.npz`` (``data/hands_box`` for boxes) holding, on the
mocap's time grid: ``base_pos`` / ``base_quat`` (T, 2, 3/4) wrist poses (left, right), ``joint_pos`` (T, nj)
in the model's hinge order, the props, the table, the glove pressure and the mocap hand-bone positions.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import tyro
from scipy.spatial.transform import Rotation

from supermimic.data.glove_taxels import fitted_thumb_roll, flexion_alignment, flexion_check, palm_frame
from supermimic.data.sweep_mocap import estimate_table, load_human, load_pressure, load_props
from supermimic.robots import CAPSULE_RIG_XML, HAND_DATA, HAND_SHAPE, HANDS_XML

ROOT = Path(__file__).resolve().parents[2]
ASSET = HANDS_XML

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
HEEL_SPREAD = 0.70  # heel width / knuckle width, between the outer finger rays
HEEL_ALONG = 0.015  # heel points this far from the wrist toward the knuckles (m)
PALM_SPLIT = 0.40  # share of each palm ray carried by the hand body; the rest is on the metacarpal body
# per hinge kind: effort limit (N m), stiffness (N m / rad); damping gives a damping ratio ~1 on the armature
EFFORT = {"flex1": 2.0, "spread1": 1.0, "flex2": 1.0, "flex3": 0.5, "thumb_flex1": 2.0, "thumb_spread1": 2.0,
          "thumb_flex2": 1.5, "thumb_spread2": 1.0, "thumb_flex3": 0.8}
ARMATURE = 2.0e-3
WRIST_ARMATURE = {"slide": 0.05, "hinge": 2.0e-3}
# Wrist joint ranges: the clip's range plus these margins (m, rad); the middle Euler angle stays off gimbal lock.
WRIST_RANGE_MARGIN = {"slide": 0.15, "hinge": 0.6}
WRIST_RX_LIMIT = 1.45
RIG_ASSET = CAPSULE_RIG_XML
SKIN = [0.87, 0.72, 0.62, 1.0]
SKIN_R = [0.80, 0.66, 0.58, 1.0]


def local_frames(human, b: int) -> tuple[Rotation, np.ndarray]:
    """Local rotation (T) and position (T, 3) of bone b in its parent's frame."""
    p = human.parents[b]
    rp = Rotation.from_quat(human.quat[:, p], scalar_first=True)
    return rp.inv() * Rotation.from_quat(human.quat[:, b], scalar_first=True), rp.inv().apply(human.pos[:, b] - human.pos[:, p])


def hinge_axes(r: Rotation, flip: bool) -> np.ndarray:
    """(3, 3) rows: principal axes of the rotation vectors (uncentered), flexion first, mean flexion >= 0 unless
    ``flip``; the rows form a right-handed frame."""
    rv = r.as_rotvec()
    w, v = np.linalg.eigh(rv.T @ rv / len(rv))
    a = v[:, ::-1].T.copy()
    if ((rv @ a[0]).mean() < 0) != flip:
        a[0] = -a[0]
    a[2] = np.cross(a[1], a[0])
    return a


def decompose(r: Rotation, axes: np.ndarray) -> tuple[np.ndarray, float]:
    """Angles (T, 3) with r = R(a0, t0) R(a1, t1) R(a2, t2) (intrinsic), and the max |t2| in degrees."""
    C = np.stack([axes[2], axes[1], axes[0]], 1)  # columns x', y', z' = a2, a1, a0
    Rc = Rotation.from_matrix(C)
    e = (Rc.inv() * r * Rc).as_euler("ZYX")  # intrinsic z' (a0), y' (a1), x' (a2)
    return e, float(np.degrees(np.abs(e[:, 2]).max()))


def add_segment(body, name: str, a: np.ndarray, b: np.ndarray, radius: float, rgba, palm: np.ndarray, shape: str,
                shorten_end: float = 0.0, side: np.ndarray | None = None) -> None:
    """A capsule from ``a`` to ``b``, or (``shape`` box) the box of the same span (``radius`` past each end) with a
    2 ``radius`` square cross-section and one face toward ``palm`` (body frame): the box's x axis; ``side``: its y
    axis points that way (the box's z then runs either way along the bone)."""
    d = b - a
    L = np.linalg.norm(d)
    b = a + d * max(L - shorten_end, 1e-3) / L
    kw = dict(rgba=rgba, density=1000.0, condim=4, friction=[1.0, 0.005, 0.0001])
    if shape == "capsule":
        body.add_geom(name=f"{name}_collision", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[*a, *b], size=[radius, 0, 0], **kw)
        return
    z = (b - a) / np.linalg.norm(b - a)
    x = palm - (palm @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    if side is not None and y @ side < 0:
        z, y = -z, -y
    quat = Rotation.from_matrix(np.stack([x, y, z], 1)).as_quat(scalar_first=True)
    body.add_geom(name=f"{name}_collision", type=mujoco.mjtGeom.mjGEOM_BOX, pos=0.5 * (a + b), quat=quat,
                  size=[radius, radius, 0.5 * np.linalg.norm(b - a) + radius], **kw)


def segment_palm_side(flex_axis: np.ndarray, a: np.ndarray, b: np.ndarray, palm: np.ndarray) -> np.ndarray:
    """Palm side of a phalanx (body frame): where positive flexion moves it (``glove_taxels.segment_palm_side``),
    up to sign (a square box is the same either way); the hand's palm normal when the hinge is near the bone axis."""
    u = (b - a) / np.linalg.norm(b - a)
    s = np.cross(flex_axis, u)
    return s if np.linalg.norm(s) > 0.5 else palm


def make_spec(human, flip: set[str], rig: dict[str, tuple[np.ndarray, np.ndarray]] | None = None, shape: str = "capsule",
              thumb_roll: float = 0.0, thumb_frames: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
              ) -> tuple[mujoco.MjSpec, dict[str, np.ndarray], list]:
    """The two hands; ``flip``: flexion hinges whose mean-flexion sign is reversed; ``shape``: segment geometry;
    ``thumb_frames`` (boxes, ``thumb_frames()``): side -> the distal thumb box's (face, side) in its body frame, built
    at roll angle ``thumb_roll`` (rad, stored as the model's ``thumb_roll`` numeric, read by glove_taxels).

    Without ``rig`` each hand is a free body. With ``rig`` (side -> wrist origin, wrist orientation) both hang from
    one root body ``rig`` (a free joint welded to the world, so the model keeps a floating root) through six wrist
    joints each, one per body: slides along world x, y, z, then hinges about the wrist's own y, x, z axes, all zero
    at the given wrist pose."""
    idx = human.index
    spec = mujoco.MjSpec()
    spec.modelname = f"{shape}_rig" if rig else f"{shape}_hands"
    spec.compiler.degree = False
    if thumb_frames:
        spec.add_numeric(name="thumb_roll", data=[thumb_roll])
    joint_angles: dict[str, np.ndarray] = {}
    report = []

    def child_pos(name: str) -> np.ndarray:
        return local_frames(human, idx(name))[1].mean(0)

    def light_body(parent, name: str, **kw):
        b = parent.add_body(name=name, **kw)
        b.explicitinertial, b.mass, b.inertia = True, 1e-3, [1e-7, 1e-7, 1e-7]
        b.gravcomp = 1.0
        return b

    root = None
    if rig:
        root = spec.worldbody.add_body(name="rig")
        root.add_freejoint(name="rig_root")
        root.explicitinertial, root.mass, root.inertia, root.gravcomp = True, 1.0, [1e-2, 1e-2, 1e-2], 1.0
        spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name="rig_weld", name1="rig", objtype=mujoco.mjtObj.mjOBJ_BODY)

    for S, side in SIDES.items():
        rgba = SKIN if S == "L" else SKIN_R
        if rig:
            p0, q0 = rig[S]
            parent = light_body(root, f"{S}_wrist_x", pos=p0)
            parent.add_joint(name=f"{S}_wrist_x", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0], armature=WRIST_ARMATURE["slide"])
            for ax, n in (([0, 1, 0], "y"), ([0, 0, 1], "z")):
                parent = light_body(parent, f"{S}_wrist_{n}")
                parent.add_joint(name=f"{S}_wrist_{n}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=ax, armature=WRIST_ARMATURE["slide"])
            parent = light_body(parent, f"{S}_wrist_ry", quat=q0)
            parent.add_joint(name=f"{S}_wrist_ry", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], armature=WRIST_ARMATURE["hinge"])
            parent = light_body(parent, f"{S}_wrist_rx")
            parent.add_joint(name=f"{S}_wrist_rx", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], armature=WRIST_ARMATURE["hinge"])
            hand = parent.add_body(name=f"{S}_hand")
            hand.add_joint(name=f"{S}_wrist_rz", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], armature=WRIST_ARMATURE["hinge"])
        else:
            hand = spec.worldbody.add_body(name=f"{S}_hand")
            hand.add_freejoint(name=f"{S}_hand_root")
        hand.add_site(name=f"{S}_wrist", size=[0.004, 0, 0])
        zero = np.zeros(3)
        # Palm: one ray per finger from a heel point to its knuckle. The mocap's metacarpal bones start within ~2 cm
        # of each other at the wrist; the heel points spread them to HEEL_SPREAD of the knuckle spread, the width of
        # the flesh at the heel of a real hand.
        mc_frame, knuckle = {}, {}
        for f in FINGERS:
            r0, lp0 = local_frames(human, idx(f"{side}InHand{f}"))
            R0 = Rotation.from_rotvec(r0.as_rotvec().mean(0))
            mc_frame[f] = (lp0.mean(0), R0)
            knuckle[f] = lp0.mean(0) + R0.apply(child_pos(f"{side}Hand{f}1"))
        along = knuckle["Middle"] / np.linalg.norm(knuckle["Middle"])
        radial = knuckle["Index"] - knuckle["Pinky"]
        radial = radial - (radial @ along) * along
        radial /= np.linalg.norm(radial)
        k_mid = np.mean([knuckle[f] for f in FINGERS], 0)
        heel = {f: HEEL_ALONG * along + HEEL_SPREAD * ((knuckle[f] - k_mid) @ radial) * radial for f in FINGERS}
        # Out of the palm (as glove_taxels.palm_frame): the left hand mirrors the right.
        palm = np.cross(radial, along) if S == "R" else np.cross(along, radial)
        palm /= np.linalg.norm(palm)

        def segment(body, name, a, b, radius, palm_side, shorten_end=0.0, side=None):
            add_segment(body, name, a, b, radius, rgba, palm_side, shape, shorten_end, side)

        segment(hand, f"{S}_heel", heel["Index"], heel["Pinky"], WRIST_RADIUS, palm)
        segment(hand, f"{S}_thenar", heel["Index"], child_pos(f"{side}HandThumb1"), THENAR_RADIUS, palm)

        def chain(parent_body, bones: list[str], joints: dict, radii, prefix: str, tip: str, palm_local: np.ndarray,
                  face_flexion: bool) -> None:
            """``face_flexion``: boxes face the segment's flexion direction (the thumb) instead of the palm."""
            body_parent = parent_body
            for k, bone in enumerate(bones):
                r, lp = local_frames(human, idx(bone))
                b = body_parent.add_body(name=f"{prefix}{k + 1}", pos=lp.mean(0))
                n_h, flex_max = joints[str(k + 1)]
                axes = hinge_axes(r, f"{prefix}{k + 1}_flex" in flip)
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
                face = segment_palm_side(axes[0], zero, end, palm_local) if face_flexion else palm_local
                side = None
                if thumb_frames and f"{prefix}{k + 1}" == f"{S}_thumb3":
                    face, side = thumb_frames[S]
                segment(b, f"{prefix}{k + 1}", zero, end, radii[k], face, shorten_end=radii[k] if nxt == tip else 0.0, side=side)
                if nxt == tip:
                    b.add_site(name=f"{prefix}_tip", pos=end, size=[0.004, 0, 0])
                body_parent = b

        for f in FINGERS:
            fl = f.lower()
            p0, R0 = mc_frame[f]
            mc = hand.add_body(name=f"{S}_{fl}0", pos=p0, quat=R0.as_quat(scalar_first=True))
            mid = heel[f] + PALM_SPLIT * (knuckle[f] - heel[f])
            palm_mc = R0.inv().apply(palm)
            segment(hand, f"{S}_{fl}_wrist", heel[f], mid, WRIST_RADIUS, palm)
            segment(mc, f"{S}_{fl}0", R0.inv().apply(mid - p0), child_pos(f"{side}Hand{f}1"), METACARPAL_RADIUS[f], palm_mc)
            chain(mc, [f"{side}Hand{f}{k}" for k in (1, 2, 3)], FINGER_JOINTS, FINGER_RADIUS[f], f"{S}_{fl}", f"{side}Hand{f}3_End",
                  palm_mc, False)
        chain(hand, [f"{side}HandThumb{k}" for k in (1, 2, 3)], THUMB_JOINTS, THUMB_RADIUS, f"{S}_thumb", f"{side}HandThumb3_End",
              palm, True)

    # Phalanges and palm pieces of one hand never collide with each other (their capsules meet at the knuckles).
    for S in SIDES:
        bodies = [b.name for b in spec.bodies if b.name.startswith(f"{S}_")]
        for i, b1 in enumerate(bodies):
            for b2 in bodies[i + 1 :]:
                spec.add_exclude(bodyname1=b1, bodyname2=b2)
    if rig:
        # The arm carries the hand's weight; the actuators are configured by the training environment.
        for b in spec.bodies:
            if b.name[:2] in ("L_", "R_"):
                b.gravcomp = 1.0
        return spec, joint_angles, report

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
    return spec, joint_angles, report


def flexion_flips(model: mujoco.MjModel, joint_angles: dict[str, np.ndarray], step: float = 0.05) -> set[str]:
    """Flexion hinges whose positive direction opens the hand (``flexion_alignment`` < 0), at the clip's median pose."""
    data = median_pose(model, joint_angles)
    return {n for n in joint_angles if n.endswith("_flex") and flexion_alignment(model, data, n) < 0}


def thumb_frames(model: mujoco.MjModel, joint_angles: dict[str, np.ndarray], roll: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Side -> (face, side) of the distal thumb box in its body frame. At the clip's median pose, across the bone, the
    face is turned ``roll`` (rad) from the direction toward the fingers (away from the index side) toward the palm
    front, and side is 90 deg further. Both are anatomical, so the two thumbs mirror each other; the thumb tip's own
    hinge axis is not used (the recording moves that joint by a few degrees only)."""
    data = median_pose(model, joint_angles)
    out = {}
    for S in SIDES:
        _, pn, radial, _ = palm_frame(model, data, S)
        b = model.body(f"{S}_thumb3").id
        u = data.site_xpos[model.site(f"{S}_thumb_tip").id] - data.xpos[b]
        u /= np.linalg.norm(u)
        e1 = -radial - (-radial @ u) * u
        e1 /= np.linalg.norm(e1)
        e2 = pn - (pn @ u) * u - (pn @ e1) * e1
        e2 /= np.linalg.norm(e2)
        R = data.xmat[b].reshape(3, 3)
        out[S] = (R.T @ (np.cos(roll) * e1 + np.sin(roll) * e2), R.T @ (np.cos(roll) * e2 - np.sin(roll) * e1))
    return out


def median_pose(model: mujoco.MjModel, joint_angles: dict[str, np.ndarray]) -> mujoco.MjData:
    data = mujoco.MjData(model)
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            data.qpos[model.jnt_qposadr[j] + 3] = 1.0
    for n, q in joint_angles.items():
        data.qpos[model.jnt_qposadr[model.joint(n).id]] = np.median(q)
    mujoco.mj_kinematics(model, data)
    return data


def main(mocap_dir: Path = ROOT / "data/sweep_mocap", clip: str = "sweep_task023_hands", out_dir: Path = HAND_DATA,
         asset: Path = ASSET, rig_asset: Path = RIG_ASSET, shape: str = HAND_SHAPE, thumb_roll_deg: float | None = None) -> None:
    """Defaults follow ``SM_HAND_SHAPE`` (supermimic.robots); ``thumb_roll_deg`` (boxes): the distal thumb box's
    fixed roll angle, default the one fitted by scripts/data/fit_glove_blocks.py."""
    thumb_roll = np.radians(thumb_roll_deg) if thumb_roll_deg is not None else fitted_thumb_roll()
    if shape == "box":
        print(f"thumb roll {np.degrees(thumb_roll):.1f} deg")
    human = load_human(mocap_dir)
    props = load_props(mocap_dir)
    pressure = load_pressure(mocap_dir)
    T = len(human.times)
    idx = human.index

    spec, joint_angles, _ = make_spec(human, set(), shape=shape)
    flips = flexion_flips(spec.compile(), joint_angles)
    spec, joint_angles, report = make_spec(human, flips, shape=shape)
    assert not flexion_flips(spec.compile(), joint_angles), "flexion signs did not settle"
    frames = thumb_frames(spec.compile(), joint_angles, thumb_roll) if shape == "box" else None
    if frames:
        spec, joint_angles, report = make_spec(human, flips, shape=shape, thumb_roll=thumb_roll, thumb_frames=frames)
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

    # The rig: wrists on 6 joints each, zeroed at each wrist's mean pose over the clip.
    rig_pose = {}
    for k, S in enumerate(SIDES):
        r = Rotation.from_quat(base_quat[:, k], scalar_first=True)
        rig_pose[S] = (base_pos[:, k].mean(0), r.mean().as_quat(scalar_first=True))
    wrist = {}
    for k, S in enumerate(SIDES):
        p0, q0 = rig_pose[S]
        rel = Rotation.from_quat(q0, scalar_first=True).inv() * Rotation.from_quat(base_quat[:, k], scalar_first=True)
        ang = np.unwrap(rel.as_euler("YXZ"), axis=0)  # intrinsic: R = Ry Rx Rz, matching the joint order
        mid = np.degrees(np.abs(ang[:, 1]).max())
        assert mid < 75.0, f"{S} wrist middle angle reaches {mid:.0f} deg: too close to gimbal lock"
        for i, n in enumerate("xyz"):
            wrist[f"{S}_wrist_{n}"] = base_pos[:, k, i] - p0[i]
        for i, n in enumerate(("ry", "rx", "rz")):
            wrist[f"{S}_wrist_{n}"] = ang[:, i]
    rig_spec, _, _ = make_spec(human, flips, rig_pose, shape=shape, thumb_roll=thumb_roll, thumb_frames=frames)
    for n, q in wrist.items():
        j = rig_spec.joint(n)
        if n[-2] == "r":
            lo, hi = max(q.min() - WRIST_RANGE_MARGIN["hinge"], -np.pi), min(q.max() + WRIST_RANGE_MARGIN["hinge"], np.pi)
            if n.endswith("rx"):
                lo, hi = max(lo, -WRIST_RX_LIMIT), min(hi, WRIST_RX_LIMIT)
        else:
            lo, hi = q.min() - WRIST_RANGE_MARGIN["slide"], q.max() + WRIST_RANGE_MARGIN["slide"]
        j.limited, j.range = mujoco.mjtLimited.mjLIMITED_TRUE, [lo, hi]
    rig_model = rig_spec.compile()
    rig_asset.write_text(rig_spec.to_xml())
    rig_names = [rig_model.joint(i).name for i in range(1, rig_model.njnt)]
    rig_joint_pos = np.stack([wrist[n] if n in wrist else joint_angles[n] for n in rig_names], 1)
    rig_data = mujoco.MjData(rig_model)
    rig_data.qpos[3] = 1.0
    rig_qadr = [rig_model.jnt_qposadr[rig_model.joint(n).id] for n in rig_names]
    rig_err = np.zeros((T, len(tips)))
    for t in range(T):
        rig_data.qpos[rig_qadr] = rig_joint_pos[t]
        mujoco.mj_kinematics(rig_model, rig_data)
        for i, (site, bone) in enumerate(tips):
            rig_err[t, i] = np.linalg.norm(rig_data.site_xpos[rig_model.site(site).id] - human.pos[t, idx(bone)])

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
        rig_joint_names=np.array(rig_names), rig_joint_pos=rig_joint_pos,
        rig_wrist_pos0=np.stack([rig_pose[S][0] for S in SIDES]), rig_wrist_quat0=np.stack([rig_pose[S][1] for S in SIDES]),
    )
    print(f"wrote {rig_asset}: {len(rig_names)} joints under the welded root (6 wrist + 21 finger per hand); "
          f"fingertip replay error mean {rig_err.mean() * 1000:.1f} mm, max {rig_err.max() * 1000:.1f} mm")
    mass = {S: sum(model.body_mass[b] for b in range(model.nbody) if model.body(b).name.startswith(f"{S}_")) for S in SIDES}
    print(f"wrote {asset}: {len(hinges)} finger hinges, {model.nu} actuators, hand mass L {mass['L']:.3f} R {mass['R']:.3f} kg")
    print(f"wrote {out}: {T} frames ({human.times[-1]:.1f} s)")
    print(f"  fingertip replay error: mean {err.mean() * 1000:.1f} mm, max {err.max() * 1000:.1f} mm")
    worst = max(report, key=lambda x: x[2])
    print(f"  largest rotation left out by the hinges: {worst[2]:.1f} deg ({worst[0]})")
    print(f"  flexion signs reversed from the mean-flexion guess: {', '.join(sorted(flips)) or 'none'}")
    align = {S: flexion_check(model, median_pose(model, joint_angles), S) for S in SIDES}
    for S in SIDES:
        print(f"  {S} flexion toward the palm (cosine, median pose): " + " ".join(f"{n[2:-5]} {v:+.2f}" for n, v in align[S].items()))
    for n in hinges:
        j = model.joint(n)
        a = joint_angles[n]
        print(f"  {n:18s} data {np.degrees(a.min()):+6.1f}..{np.degrees(a.max()):+6.1f} deg   range {np.degrees(j.range[0]):+6.1f}..{np.degrees(j.range[1]):+6.1f}")
    if asset == ASSET and out.parent == HAND_DATA:
        sys.path.insert(0, str(ROOT / "scripts/media"))
        from render_glove_blocks import hands_picture

        roll = f", thumb tip roll {np.degrees(thumb_roll):+.1f} deg" if shape == "box" else ""
        hands_picture("hands_built", f"build_capsule_hands.py{roll}", clip=out, grasp=False)


if __name__ == "__main__":
    tyro.cli(main)

"""Can the retargeted grasps hold the props in physics? A check before any learning.

  SM_ROBOT=g1_inspire uv run python scripts/data/check_prop_hold.py \
      --retarget ~/g1-sweep-retarget/out/sweep_g1_inspire_closed.npz --squeeze 0 0.3 --start 0 6

The robot's root and body joints follow the reference exactly (written into the state every physics step); only the
hands and the props are simulated. Fingers are position-controlled toward the reference with SuperMimic's Inspire PD
gains, effort limit and mimic couplings; on frames where a finger's glove regions are loaded the target is closed
further by ``squeeze`` (rad), since a target exactly at the surface produces almost no grip force. The broom and the
dustpan are free bodies (CoACD convex pieces, scripts/assets/build_props.py) starting at their reference pose; the
table is a static box. Hand collision geoms get sliding and torsional friction (``condim`` 4 by default; ``torsional``
is the torsional coefficient, a length: roughly the radius of a fingertip contact patch times the friction
coefficient), unlike mjlab's G1 collision config (condim 1, frictionless).

Prints, per run and prop, how far the prop drifts from its reference trajectory; ``--video`` renders the run.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import tyro
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from close_fingers_on_contact import RETARGET_ROOT, glove_region_loaded

ROOT = Path(__file__).resolve().parents[2]
PROP_RGBA = {"cleaningtools_broom": [0.80, 0.55, 0.30, 1], "cleaningtools_dustpan": [0.30, 0.50, 0.80, 1]}
# Finger driver -> glove regions that load it.
DRIVER_REGIONS = {"thumb_proximal_pitch": ("thumb_tip", "thumb_base"), "index_proximal": ("index_tip", "index_proximal"),
                  "middle_proximal": ("middle_tip", "middle_proximal"), "ring_proximal": ("ring_tip", "ring_proximal"),
                  "pinky_proximal": ("little_tip", "little_proximal")}


def build(z: dict, props_dir: Path, masses: dict[str, float], hand_friction: float, prop_friction: float, dt: float,
          condim: int = 4, torsional: float = 0.005):
    from supermimic.robots import INSPIRE_HAND_ACTUATED, INSPIRE_HAND_ARMATURE, INSPIRE_HAND_EFFORT, ROBOT

    spec = ROBOT.spec_fn()
    spec.option.timestep = dt
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    hand_bodies = set()
    for s in "LR":
        hand_bodies.add(f"{s}_hand_base_link")
        hand_bodies |= {b.name for b in spec.body(f"{s}_hand_base_link").find_all(mujoco.mjtObj.mjOBJ_BODY)}
    for g in spec.geoms:
        hand_col = g.name.endswith("_collision") and g.parent.name in hand_bodies
        g.contype = g.conaffinity = 1 if hand_col else 0
        if hand_col:
            g.condim, g.friction = condim, [hand_friction, torsional, 0.0001]
    wn = 10 * 2.0 * math.pi
    kp, kd = INSPIRE_HAND_ARMATURE * wn**2, 2.0 * 2.0 * INSPIRE_HAND_ARMATURE * wn
    for j in INSPIRE_HAND_ACTUATED:
        spec.joint(j).armature = INSPIRE_HAND_ARMATURE
        a = spec.add_actuator(name=j, target=j, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.gainprm[0], a.biastype, a.biasprm[:3] = kp, mujoco.mjtBias.mjBIAS_AFFINE, [0.0, -kp, -kd]
        a.forcelimited, a.forcerange = True, [-INSPIRE_HAND_EFFORT, INSPIRE_HAND_EFFORT]
    for a in list(spec.actuators):
        if a.name not in INSPIRE_HAND_ACTUATED:
            spec.delete(a)

    scale = float(z["scale"])
    w = spec.worldbody
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], rgba=[0.85, 0.86, 0.88, 1], friction=[prop_friction, 0.005, 0.0001])
    w.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, pos=z["table_center"] * scale, size=z["table_half"] * scale,
               rgba=[0.62, 0.48, 0.34, 1], friction=[prop_friction, 0.005, 0.0001])
    w.add_light(pos=[0, 0, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.6, 0.6, 0.6])
    names = [str(n) for n in z["prop_names"]][:2]
    for n in names:
        b = w.add_body(name=n)
        b.add_freejoint(name=f"{n}_free")
        pieces = sorted((props_dir / n).glob("collision_*.obj"))
        for i, f in enumerate(pieces):
            spec.add_mesh(name=f"{n}_c{i:02d}", file=str(f.resolve()), scale=[scale] * 3)
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{n}_c{i:02d}", group=3, friction=[prop_friction, 0.005, 0.0001],
                       density=1.0)
        spec.add_mesh(name=f"{n}_visual", file=str((props_dir / n / "visual.stl").resolve()), scale=[scale] * 3)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{n}_visual", contype=0, conaffinity=0, group=2, rgba=PROP_RGBA[n], density=0.0)
    model = spec.compile()
    # Density 1 gives each prop its hull volume as mass; rescale to the requested mass (inertia scales with it).
    for n in names:
        bid = model.body(n).id
        k = masses[n] / model.body_mass[bid]
        model.body_mass[bid] *= k
        model.body_inertia[bid] *= k
    return model, names


def quat_interp(q: np.ndarray, t: np.ndarray, tq: float) -> np.ndarray:
    i = int(np.clip(np.searchsorted(t, tq) - 1, 0, len(t) - 2))
    a = (tq - t[i]) / (t[i + 1] - t[i])
    q0, q1 = q[i], q[i + 1] * (1 if np.dot(q[i], q[i + 1]) >= 0 else -1)
    out = (1 - a) * q0 + a * q1
    return out / np.linalg.norm(out)


def main(retarget: Path, props_dir: Path = ROOT / "data/props", retarget_root: Path = RETARGET_ROOT,
         squeeze: tuple[float, ...] = (0.0, 0.3), start: tuple[float, ...] = (0.0, 6.0), seconds: float | None = None,
         broom_mass: float = 0.10, dustpan_mass: float = 0.20, hand_friction: float = 1.0, prop_friction: float = 0.8,
         condim: int = 4, torsional: float = 0.005,
         dt: float = 0.005, rest: tuple[float, float] = (0.0, 2.0), video: Path | None = None, fps: int = 30,
         focus: str = "both", cam_distance: float = 1.3, cam_azimuth: float = 200.0, cam_elevation: float = -25.0) -> None:
    """``start``: reference times (s) to start from; ``seconds``: run length (default: to the clip end);
    ``video``: mp4 path prefix, one file per run; ``focus``: both | left | right hand; ``cam_azimuth``: degrees
    relative to the direction the robot faces."""
    from supermimic.robots import INSPIRE_HAND_ACTUATED, ROBOT

    assert ROBOT.name == "g1_inspire", "run with SM_ROBOT=g1_inspire"
    sys.path.insert(0, str(retarget_root))
    from sweep_retarget.pressure import REGIONS

    z = dict(np.load(retarget))
    assert bool(z.get("props_grounded", False)), "retarget predates grounded props: re-run retarget.py"
    masses = {"cleaningtools_broom": broom_mass, "cleaningtools_dustpan": dustpan_mass}
    model, props = build(z, props_dir, masses, hand_friction, prop_friction, dt, condim, torsional)
    data = mujoco.MjData(model)
    times, qref = z["times"], z["qpos"]
    jn = [str(n) for n in z["joint_names"]]
    hand = np.array([n[:2] in ("L_", "R_") for n in jn])
    body_qadr = np.array([0, 1, 2] + [model.jnt_qposadr[model.joint(n).id] for n, h in zip(jn, hand) if not h])
    body_jcols = np.array([7 + i for i, h in enumerate(hand) if not h])
    body_dadr = np.array([model.jnt_dofadr[model.joint(n).id] for n, h in zip(jn, hand) if not h])
    act_qcol = np.array([7 + jn.index(a) for a in INSPIRE_HAND_ACTUATED])
    prop_qadr = [model.jnt_qposadr[model.joint(f"{n}_free").id] for n in props]
    prop_dadr = [model.jnt_dofadr[model.joint(f"{n}_free").id] for n in props]

    region_on = glove_region_loaded(z, retarget_root, rest)
    reg = list(REGIONS)
    act_on = np.zeros((len(times), len(INSPIRE_HAND_ACTUATED)), bool)
    for k, a in enumerate(INSPIRE_HAND_ACTUATED):
        drv = a[2:].removesuffix("_joint")
        if drv in DRIVER_REGIONS:
            side = "left" if a[0] == "L" else "right"
            act_on[:, k] = region_on[side][:, [reg.index(r) for r in DRIVER_REGIONS[drv]]].any(1)

    def ref_at(tq: float):
        q = np.array([np.interp(tq, times, qref[:, c]) for c in range(qref.shape[1])])
        q[3:7] = quat_interp(qref[:, 3:7], times, tq)
        return q

    def ref_prop(tq: float, j: int):
        p = np.array([np.interp(tq, times, z["prop_pos"][:, j, c]) for c in range(3)])
        return p, quat_interp(z["prop_quat"][:, j], times, tq)

    renderer = cam = None
    if video is not None:
        renderer = mujoco.Renderer(model, 720, 1280)
        cam = mujoco.MjvCamera()
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        opt.geomgroup[0] = opt.geomgroup[2] = 1
        yaw = Rotation.from_quat(qref[:, 3:7], scalar_first=True).as_euler("zyx")[:, 0]
        facing = math.degrees(math.atan2(np.sin(yaw).mean(), np.cos(yaw).mean()))

    for t0 in start:
        t1 = times[-1] if seconds is None else min(times[-1], t0 + seconds)
        for sq in squeeze:
            mujoco.mj_resetData(model, data)
            data.qpos[: qref.shape[1]] = ref_at(t0)
            h = 1e-3
            for j, n in enumerate(props):
                p, q = ref_prop(t0, j)
                p2, q2 = ref_prop(t0 + h, j)
                data.qpos[prop_qadr[j] : prop_qadr[j] + 3], data.qpos[prop_qadr[j] + 3 : prop_qadr[j] + 7] = p, q
                data.qvel[prop_dadr[j] : prop_dadr[j] + 3] = (p2 - p) / h
                rv = (Rotation.from_quat(q, scalar_first=True).inv() * Rotation.from_quat(q2, scalar_first=True)).as_rotvec()
                data.qvel[prop_dadr[j] + 3 : prop_dadr[j] + 6] = rv / h
            mujoco.mj_forward(model, data)
            steps = int(round((t1 - t0) / dt))
            err = np.zeros((steps, len(props)))
            ang = np.zeros((steps, len(props)))
            ff = None
            if renderer is not None:
                out = video.with_name(f"{video.stem}_t{t0:g}_sq{sq:g}.mp4")
                out.parent.mkdir(parents=True, exist_ok=True)
                ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720", "-r", str(fps), "-i", "-",
                                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)], stdin=subprocess.PIPE)
                every = max(1, int(round(1 / (fps * dt))))
                target = None
            for k in range(steps):
                tq = t0 + (k + 1) * dt
                q, qn = ref_at(tq - dt), ref_at(tq)
                data.qpos[body_qadr[:3]] = qn[:3]
                data.qpos[3:7] = qn[3:7]
                data.qpos[body_qadr[3:]] = qn[body_jcols]
                data.qvel[0:3] = (qn[:3] - q[:3]) / dt
                rq = Rotation.from_quat(q[3:7], scalar_first=True)
                data.qvel[3:6] = (rq.inv() * Rotation.from_quat(qn[3:7], scalar_first=True)).as_rotvec() / dt
                data.qvel[body_dadr] = (qn[body_jcols] - q[body_jcols]) / dt
                fi = int(np.clip(np.searchsorted(times, tq), 0, len(times) - 1))
                data.ctrl[:] = qn[act_qcol] + sq * act_on[fi]
                mujoco.mj_step(model, data)
                for j in range(len(props)):
                    p, qr = ref_prop(tq, j)
                    err[k, j] = np.linalg.norm(data.qpos[prop_qadr[j] : prop_qadr[j] + 3] - p)
                    qs = data.qpos[prop_qadr[j] + 3 : prop_qadr[j] + 7]
                    ang[k, j] = 2 * math.degrees(math.acos(min(1.0, abs(float(np.dot(qs, qr))))))
                if ff is not None and k % every == 0:
                    hands = {"left": ["L_hand_base_link"], "right": ["R_hand_base_link"], "both": ["L_hand_base_link", "R_hand_base_link"]}[focus]
                    f = data.xpos[[model.body(b).id for b in hands]].mean(0)
                    target = f if target is None else 0.9 * target + 0.1 * f
                    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = target, cam_distance, facing + cam_azimuth, cam_elevation
                    renderer.update_scene(data, cam, opt)
                    ff.stdin.write(renderer.render().tobytes())
            if ff is not None:
                ff.stdin.close()
                ff.wait()
            print(f"start {t0:4.1f} s, {t1 - t0:4.1f} s run, squeeze {sq:.2f} rad")
            for j, n in enumerate(props):
                e = err[:, j] * 100
                drop = np.flatnonzero(e > 10.0)
                print(f"  {n:24s} position error median {np.median(e):5.1f} cm, p90 {np.percentile(e, 90):5.1f} cm, "
                      f"within 5 cm {np.mean(e < 5):4.0%}; angle median {np.median(ang[:, j]):4.0f} deg; "
                      f"first > 10 cm at {'never' if not len(drop) else f'{t0 + (drop[0] + 1) * dt:.2f} s'}")


if __name__ == "__main__":
    tyro.cli(main)

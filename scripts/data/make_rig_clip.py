"""50 Hz SuperMimic clip of the capsule-hand rig (SM_ROBOT=capsule_hands) from a hands npz of build_capsule_hands.py.

  SM_ROBOT=capsule_hands uv run python scripts/data/make_rig_clip.py \
      --src data/hands/sweep_task023_hands_registered_closed.npz --out data/npz/capsule_hands/sweep_task023.npz

The wrist joints are recomputed from the source's wrist poses (the registration and finger-closure steps move the
wrists and fingers without updating ``rig_joint_*``): slides = wrist position - rig_wrist_pos0, hinges = the
intrinsic y-x-z angles of rig_wrist_quat0^-1 * wrist orientation. Positions are interpolated linearly, rotations
by slerp; velocities by central differences. Body link states come from MuJoCo FK of the rig.

Also written, for the scene with props: the broom and dustpan reference (prop_*_w), the table, scene_scale 1
(the hands are at human scale), ``hand_loaded`` (all False: the grip reward reads the glove itself) and
``taxel_load`` (T, 2, 129), the glove pressure above each taxel's rest baseline (raw glove units, left, right;
sweep_mocap.taxel_load), readings below ``load_floor`` set to 0. A hand holds on the frames where its total pressure
exceeds ``hold_rel`` of its clip 95th percentile (and ``hold_floor``), in holds of at least ``min_hold`` seconds;
outside them its pressure is set to 0 (the resting fit of a free glove).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import tyro
from scipy.spatial.transform import Rotation, Slerp

PROP_SOURCES = {"broom": "cleaningtools_broom", "dustpan": "cleaningtools_dustpan"}
ROOT = Path(__file__).resolve().parents[2]


def main(src: Path = ROOT / "data/hands/sweep_task023_hands_registered_closed.npz",
         out: Path = ROOT / "data/npz/capsule_hands/sweep_task023.npz", fps: float = 50.0, min_hold: float = 0.5,
         load_floor: float = 3.0, hold_rel: float = 0.35, hold_floor: float = 30.0) -> None:
    from supermimic.data.motion_convert import ResampledMotion, _so3_derivative, forward_kinematics
    from supermimic.robots import ROBOT

    from supermimic.data.sweep_mocap import taxel_load

    assert ROBOT.name.startswith("capsule_hands"), "run with SM_ROBOT=capsule_hands"
    z = np.load(src)
    times = z["times"]
    t = np.arange(times[0], times[-1] + 1e-9, 1.0 / fps)
    T = len(t)
    dt = 1.0 / fps
    lerp = lambda a: np.stack([np.interp(t, times, a[:, i]) for i in range(a.shape[1])], 1)  # noqa: E731
    slerp = lambda q: Slerp(times, Rotation.from_quat(q, scalar_first=True))(t)  # noqa: E731

    joints: dict[str, np.ndarray] = {}
    wrist_pos, wrist_rot = {}, {}
    for k, S in enumerate("LR"):
        p0, q0 = z["rig_wrist_pos0"][k], z["rig_wrist_quat0"][k]
        wrist_pos[S] = lerp(z["base_pos"][:, k])
        wrist_rot[S] = slerp(z["base_quat"][:, k])
        ang = np.unwrap((Rotation.from_quat(q0, scalar_first=True).inv() * wrist_rot[S]).as_euler("YXZ"), axis=0)
        for i, a in enumerate("xyz"):
            joints[f"{S}_wrist_{a}"] = wrist_pos[S][:, i] - p0[i]
        for i, a in enumerate(("ry", "rx", "rz")):
            joints[f"{S}_wrist_{a}"] = ang[:, i]
    fingers = lerp(z["joint_pos"])
    for i, n in enumerate(str(n) for n in z["joint_names"]):
        joints[n] = fingers[:, i]
    joint_pos = np.stack([joints[n] for n in ROBOT.joint_names], 1)

    model = ROBOT.mj_model()
    lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
    outside = (joint_pos < lo - 1e-6) | (joint_pos > hi + 1e-6)
    assert not outside.any(), f"reference outside the joint ranges: {[ROBOT.joint_names[j] for j in np.where(outside.any(0))[0]]}"

    motion = ResampledMotion(
        fps=fps, root_pos=np.zeros((T, 3)), root_quat=np.tile([1.0, 0.0, 0.0, 0.0], (T, 1)), joint_pos=joint_pos,
        root_lin_vel_w=np.zeros((T, 3)), root_ang_vel_w=np.zeros((T, 3)), joint_vel=np.gradient(joint_pos, dt, axis=0),
    )
    bodies = forward_kinematics(model, motion)

    # The FK wrists must land on the source's wrist poses.
    err_p = max(np.abs(bodies.pos_w[:, bodies.body_names.index(f"{S}_hand")] - wrist_pos[S]).max() for S in "LR")
    err_r = max(np.degrees((Rotation.from_quat(bodies.quat_w[:, bodies.body_names.index(f"{S}_hand")], scalar_first=True).inv()
                            * wrist_rot[S]).magnitude()).max() for S in "LR")
    assert err_p < 1e-5 and err_r < 1e-3, (err_p, err_r)

    src_names = [str(n) for n in z["prop_names"]]
    pos, quat, lin, ang = [], [], [], []
    for name in ("broom", "dustpan"):
        j = src_names.index(PROP_SOURCES[name])
        p, q = lerp(z["prop_pos"][:, j]), slerp(z["prop_quat"][:, j]).as_quat(scalar_first=True)
        pos.append(p)
        quat.append(q)
        lin.append(np.gradient(p, dt, axis=0))
        ang.append(_so3_derivative(q, dt))

    load = np.stack([taxel_load(z[f"pressure_{s}"], z["pressure_times"], t) for s in ("left", "right")], 1)
    load = np.where(load > load_floor, load, 0.0)
    total = load.sum(2)  # (T, 2)
    hold = total > np.maximum(hold_rel * np.percentile(total, 95, axis=0), hold_floor)
    for k in range(2):
        edge = np.diff(np.r_[0, hold[:, k].astype(int), 0])
        for a, b in zip(np.nonzero(edge == 1)[0], np.nonzero(edge == -1)[0]):
            if b - a < min_hold * fps:
                hold[a:b, k] = False
    load *= hold[..., None]

    f32 = lambda a: np.asarray(a, np.float32)  # noqa: E731
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out, fps=np.array([fps], np.float32), joint_pos=f32(joint_pos), joint_vel=f32(motion.joint_vel),
        body_pos_w=f32(bodies.pos_w), body_quat_w=f32(bodies.quat_w), body_lin_vel_w=f32(bodies.lin_vel_w), body_ang_vel_w=f32(bodies.ang_vel_w),
        joint_names=np.array(ROBOT.joint_names), body_names=np.array(bodies.body_names), source=np.array(str(src)),
        input_fps=np.array([1.0 / np.median(np.diff(times))], np.float32), times=f32(t),
        prop_names=np.array(["broom", "dustpan"]), prop_source_names=np.array([PROP_SOURCES[n] for n in ("broom", "dustpan")]),
        prop_pos_w=f32(np.stack(pos, 1)), prop_quat_w=f32(np.stack(quat, 1)), prop_lin_vel_w=f32(np.stack(lin, 1)), prop_ang_vel_w=f32(np.stack(ang, 1)),
        table_center=f32(z["table_center"]), table_half=f32(z["table_half"]), scene_scale=np.array(1.0),
        hand_loaded=np.zeros((T, len(ROBOT.hand_joint_names)), bool), hand_loaded_joints=np.array(ROBOT.hand_joint_names),
        taxel_load=f32(load),
    )
    for k, S in enumerate("LR"):
        g = np.nonzero(hold[:, k])[0]
        print(f"  {S} glove: holding {t[g[0]]:.2f}-{t[g[-1]]:.2f} s ({len(g)} frames), total pressure median {np.median(total[g, k]):.0f} raw, "
              f"{(load[g, k] > 0).sum(1).mean():.1f} loaded taxels per frame")
    speed = {S: np.linalg.norm(np.gradient(wrist_pos[S], dt, axis=0), axis=1) for S in "LR"}
    print(f"wrote {out}: {T} frames ({(T - 1) / fps:.1f} s at {fps:g} Hz), FK wrist error {err_p * 1000:.3f} mm / {err_r:.4f} deg")
    for S in "LR":
        print(f"  {S} wrist speed median {np.median(speed[S]):.2f} max {speed[S].max():.2f} m/s, "
              f"travel {np.ptp(wrist_pos[S], 0).round(3)} m")


if __name__ == "__main__":
    tyro.cli(main)

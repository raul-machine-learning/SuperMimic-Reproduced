"""How jerky a capsule-hand rollout is, against the mocap reference it tracks.

  SM_ROBOT=capsule_hands_props uv run python scripts/diag_smoothness.py --rollouts media/capsule_grip_s3_it7000_pressure.npz ...

From scripts/media/record_rollout.py rollouts (50 Hz joint positions of the robot and of the reference), over each
rollout's own length: per hand, the RMS wrist jerk (third derivative of the hand body's position, m/s^3), the RMS
wrist angular acceleration (rad/s^2), the RMS finger-joint acceleration (rad/s^2), and the share of the wrist's and
fingers' velocity power above ``cutoff`` Hz (the tremor/chatter band; mocap hand motion sits well below it).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import tyro
from scipy.spatial.transform import Rotation

DT = 0.02


def high_band(v: np.ndarray, cutoff: float) -> float:
    """Share of the power of (T, k) velocities above ``cutoff`` Hz."""
    p = np.abs(np.fft.rfft(v - v.mean(0), axis=0)) ** 2
    f = np.fft.rfftfreq(len(v), DT)
    return float(p[f > cutoff].sum() / max(p.sum(), 1e-12))


def metrics(qpos: np.ndarray, model: mujoco.MjModel, cutoff: float) -> dict[str, list[float]]:
    data = mujoco.MjData(model)
    hands = [model.body(f"{S}_hand").id for S in "LR"]
    pos, quat = np.zeros((len(qpos), 2, 3)), np.zeros((len(qpos), 2, 4))
    for t, q in enumerate(qpos):
        data.qpos[:] = q[: model.nq]
        mujoco.mj_kinematics(model, data)
        pos[t], quat[t] = data.xpos[hands], data.xquat[hands]
    out: dict[str, list[float]] = {k: [] for k in ("wrist jerk m/s^3", "wrist ang acc rad/s^2", "finger acc rad/s^2", "wrist >cut", "finger >cut")}
    for k, S in enumerate("LR"):
        p = pos[:, k]
        jerk = np.diff(p, 3, axis=0) / DT**3
        R = Rotation.from_quat(quat[:, k], scalar_first=True)
        w = (R[:-1].inv() * R[1:]).as_rotvec() / DT  # body-frame angular velocity
        fj = [model.jnt_qposadr[j] for j in range(model.njnt) if model.joint(j).name.startswith(f"{S}_")
              and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and "wrist" not in model.joint(j).name]
        fq = qpos[:, fj]
        out["wrist jerk m/s^3"].append(float(np.sqrt((np.linalg.norm(jerk, axis=-1) ** 2).mean())))
        out["wrist ang acc rad/s^2"].append(float(np.sqrt((np.linalg.norm(np.diff(w, axis=0) / DT, axis=-1) ** 2).mean())))
        out["finger acc rad/s^2"].append(float(np.sqrt(((np.diff(fq, 2, axis=0) / DT**2) ** 2).mean())))
        out["wrist >cut"].append(high_band(np.diff(p, axis=0) / DT, cutoff))
        out["finger >cut"].append(high_band(np.diff(fq, axis=0) / DT, cutoff))
    return out


def main(rollouts: list[Path], cutoff: float = 5.0) -> None:
    from supermimic.robots import ROBOT

    model = ROBOT.mj_model()
    rows = []
    for f in rollouts:
        r = np.load(f)
        T = int(r["lengths"][0])
        if not rows:
            rows.append(("mocap reference", metrics(r["ref_qpos"][0, :T], model, cutoff), T))
        rows.append((f.stem, metrics(r["robot_qpos"][0, :T], model, cutoff), T))
    keys = list(rows[0][1])
    print(f"{'rollout':38s} {'frames':>6s} " + " ".join(f"{k.replace('>cut', f'>{cutoff:g}Hz'):>24s}" for k in keys))
    for name, m, T in rows:
        print(f"{name:38s} {T:6d} " + " ".join(f"{'L ' + format(m[k][0], '.3g') + '  R ' + format(m[k][1], '.3g'):>24s}" for k in keys))


if __name__ == "__main__":
    tyro.cli(main)

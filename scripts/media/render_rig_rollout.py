"""Video of a capsule-hand rig rollout (scripts/media/record_rollout.py with SM_ROBOT=capsule_hands).

  SM_ROBOT=capsule_hands MUJOCO_GL=egl uv run python scripts/media/render_rig_rollout.py \
      --rollout media/capsule_air_rollout.npz --out media/capsule_air.webm

The rig's joint positions go through MuJoCo FK to the wrist poses and finger angles that
scripts/media/render_capsule_hands.py draws. Simulated props (``prop_qpos`` in the rollout) are drawn solid and the
mocap props translucent; without simulated props only the translucent mocap props are shown.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_capsule_hands import main as render

ROOT = Path(__file__).resolve().parents[2]


def main(rollout: Path, out: Path, clip: Path = ROOT / "data/hands/sweep_task023_hands_registered_closed.npz", index: int = 0,
         title: str = "Capsule hands: learned tracking policy", subtitle: str = "physics, hands in the air", fps: int = 30,
         seconds: float | None = None) -> None:
    """``seconds``: render only the first part of the rollout. Glove taxels are drawn on the simulated hands, colored by
    the simulated pressure when the rollout has it (record_rollout.py with a grip robot)."""
    from supermimic.robots import ROBOT

    assert ROBOT.name.startswith("capsule_hands"), "run with SM_ROBOT=capsule_hands or capsule_hands_props"
    r = np.load(rollout)
    c = dict(np.load(clip))
    T = int(r["lengths"][index]) if seconds is None else min(int(r["lengths"][index]), int(seconds * 50) + 1)
    qpos = r["robot_qpos"][index, :T]
    model = ROBOT.mj_model()
    data = mujoco.MjData(model)
    hands = [model.body(f"{S}_hand").id for S in "LR"]
    fingers = [str(n) for n in c["joint_names"]]
    fadr = [model.jnt_qposadr[model.joint(n).id] for n in fingers]
    base_pos, base_quat = np.zeros((T, 2, 3)), np.zeros((T, 2, 4))
    for t in range(T):
        data.qpos[:] = qpos[t, : model.nq]
        mujoco.mj_kinematics(model, data)
        base_pos[t], base_quat[t] = data.xpos[hands], data.xquat[hands]
    P = len(c["prop_names"])
    prop_pos = np.full((T, P, 3), 20.0)
    prop_quat = np.tile([1.0, 0.0, 0.0, 0.0], (T, P, 1))
    if "prop_qpos" in r:
        src = [str(n) for n in c["prop_names"]]
        for k, name in enumerate(str(n) for n in r["prop_source_names"]):
            prop_pos[:, src.index(name)] = r["prop_qpos"][index, :T, k, :3]
            prop_quat[:, src.index(name)] = r["prop_qpos"][index, :T, k, 3:7]
    times = c["times"][0] + np.arange(T) / 50.0
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "rollout.npz"
        extra = {}
        if "taxel_pressure" in r:
            sides = ["LR".index(s) for s, _ in ROBOT.grip]
            sim, agree, hold = np.zeros((T, 2, 129), np.float32), np.ones((T, 2), np.float32), np.zeros((T, 2), bool)
            sim[:, sides] = r["taxel_pressure"][index, :T, :, 0].astype(np.float32)
            agree[:, sides] = r["pressure_agreement"][index, :T]
            hold[:, sides] = r["ref_holding"][index, :T]
            extra = dict(taxel_sim=sim, pressure_agreement=agree, ref_holding=hold)
        np.savez(f, times=times, base_pos=base_pos, base_quat=base_quat, joint_pos=qpos[:, fadr], prop_pos=prop_pos, prop_quat=prop_quat, **extra)
        render(out=out, clip=clip, rollout=f, title=title, subtitle=subtitle, fps=fps, ghosts=True, legend="none", taxels=True, mocap_dots=False, palm_view=True)


if __name__ == "__main__":
    tyro.cli(main)

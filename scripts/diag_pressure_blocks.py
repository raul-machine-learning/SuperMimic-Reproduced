"""Glove pressure agreement per block of taxels (fingers, palm rows), over the frames where the glove holds.

  SM_ROBOT=capsule_hands_props uv run python scripts/diag_pressure_blocks.py --rollout media/capsule_grip_s3_it7000_pressure.npz
  SM_ROBOT=capsule_hands_props uv run python scripts/diag_pressure_blocks.py --mocap-pose

``--rollout``: a scripts/media/record_rollout.py rollout (``taxel_pressure``: simulated N and glove raw units per
taxel). ``--mocap-pose``: hands and props placed at the mocap pose at mid-grip frames, one physics step (as
tests/test_grip_force.py::test_mocap_pose_pressure_agreement).

Per hand and block B, averaged over holding frames, with p and q the per-hand shares of
supermimic.kinematics.glove.pressure_agreement (simulated and glove):
  glove share q_B and sim share p_B     how much of the hand's pressure falls in B
  overlap o_B = sum_{i in B} min(p, q)  B's part of the hand's score (the blocks plus the "nothing" share add up to it)
  within-block score                    pressure_agreement on B's taxels alone (1: same proportions inside B, or
                                        both press nothing there)
and the block-level score: pressure_agreement on the 10 block totals (where, at block resolution, the pressure is).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import tyro

from supermimic.data.glove_taxels import GLOVE_BASE, PALM_ROWS

BLOCKS = {"thumb": range(0, 13), **{f: range(b, b + 11) for f, b in GLOVE_BASE.items()},
          **{f"palm row {k + 1}": range(lo, hi) for k, (lo, hi) in enumerate(PALM_ROWS)}}


def mocap_pose(times: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    from sweep_data import sweep_files

    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.rewards import sim_taxel_forces
    from supermimic.envs.state_io import StateIO
    from supermimic.robots import ROBOT

    env = make_env(SuperMimicEnvCfgArgs(npz_files=sweep_files(1), num_envs=len(times), obs_noise=False, reset_noise=False,
                                        domain_randomization=False, action_noise_std=0.0, sampling_mode="start", seed=0), "cuda:0")
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        cmd.forced_frame[:] = lib.clip_start[0] + torch.tensor([round(t * 50) for t in times], device=env.device)
        env.reset()
        StateIO(env).write(torch.arange(env.num_envs, device=env.device), lib._ref[cmd.frame].float())
        env.step(torch.zeros(env.num_envs, ROBOT.na, device=env.device))
        return sim_taxel_forces(env, cmd).cpu().numpy(), cmd.taxel_load.cpu().numpy()  # (n, G, 129) each
    finally:
        env.close()


def main(rollout: Path | None = None, mocap_pose_frames: bool = False, index: int = 0,
         times: tuple[float, ...] = (3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)) -> None:
    from supermimic.kinematics.glove import GloveModel
    from supermimic.robots import ROBOT

    if mocap_pose_frames:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
        f, q = mocap_pose(times)
        source = f"mocap pose, {len(times)} mid-grip frames"
    else:
        assert rollout is not None, "--rollout or --mocap-pose-frames"
        r = np.load(rollout)
        T = int(r["lengths"][index])
        tp = r["taxel_pressure"][index, :T].astype(np.float32)  # (T, G, 2, 129)
        hold = r["ref_holding"][index, :T]  # (T, G)
        f, q = tp[:, :, 0], tp[:, :, 1]
        source = f"{rollout.name}"
    f, q = torch.as_tensor(f), torch.as_tensor(q)
    agree = GloveModel.pressure_agreement
    from supermimic.kinematics.glove import GLOVE_PRESSURE_FLOOR, SIM_PRESSURE_FLOOR

    P = f / (f.sum(-1, keepdim=True) + SIM_PRESSURE_FLOOR)
    Q = q / (q.sum(-1, keepdim=True) + GLOVE_PRESSURE_FLOOR)
    print(f"glove pressure agreement per block ({source}; frames where the glove holds)")
    for g, (side, prop) in enumerate(ROBOT.grip):
        m = torch.ones(len(f), dtype=torch.bool) if mocap_pose_frames else torch.as_tensor(hold[:, g])
        Pg, Qg, fg, qg = P[m, g], Q[m, g], f[m, g], q[m, g]
        total = agree(fg, qg).mean().item()
        bt_f = torch.stack([fg[:, list(ix)].sum(-1) for ix in BLOCKS.values()], -1)
        bt_q = torch.stack([qg[:, list(ix)].sum(-1) for ix in BLOCKS.values()], -1)
        print(f"\n{side} hand ({prop}), {int(m.sum())} frames: score {total:.3f}, block-level score {agree(bt_f, bt_q).mean().item():.3f}, "
              f"'nothing' overlap {torch.minimum(1 - Pg.sum(-1), 1 - Qg.sum(-1)).mean().item():.3f}")
        print(f"  {'block':12s} {'glove share':>11s} {'sim share':>9s} {'overlap':>7s} {'within-block':>12s}")
        for name, ix in BLOCKS.items():
            ix = list(ix)
            within = agree(fg[:, ix], qg[:, ix]).mean().item()
            print(f"  {name:12s} {Qg[:, ix].sum(-1).mean().item():11.3f} {Pg[:, ix].sum(-1).mean().item():9.3f} "
                  f"{torch.minimum(Pg[:, ix], Qg[:, ix]).sum(-1).mean().item():7.3f} {within:12.3f}")


if __name__ == "__main__":
    tyro.cli(main)

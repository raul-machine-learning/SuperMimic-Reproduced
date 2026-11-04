"""Which termination ends the episodes: replay the reference (zero residual) with the grip assist at level ``assist``,
training-like settings (random start frames, action / reset noise, domain randomisation), and count each
termination term and the clip frames where they fire.

  SM_ROBOT=capsule_hands_props uv run python scripts/diag_terminations.py --assist 1.0
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp import grip_assist
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.robots import ROBOT


@torch.no_grad()
def main(assist: float = 1.0, clip: str = "sweep_task023", npz_dir: Path = Path("data/npz/capsule_hands"), num_envs: int = 512,
         steps: int = 400, action_noise_std: float = 0.1, reset_noise: bool = True, domain_randomization: bool = True,
         device: str = "cuda:0") -> None:
    files = [str(f) for f in discover_npz(npz_dir, names=[clip])]
    args = SuperMimicEnvCfgArgs(npz_files=files, num_envs=num_envs, action_noise_std=action_noise_std, reset_noise=reset_noise,
                                domain_randomization=domain_randomization, sampling_mode="uniform", record_transitions=False, seed=0)
    env = make_env(args, device=device)
    grip_assist.set_level(assist)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        tm = env.termination_manager
        env.reset()
        ep_len = torch.zeros(num_envs, device=device)
        counts: Counter[str] = Counter()
        frames: dict[str, list[int]] = {}
        lens = []
        for _ in range(steps):
            frame_before = (cmd.frame - lib.clip_start[lib.frame_clip[cmd.frame]]).clone()
            _, _, te, to, _ = env.step(torch.zeros(num_envs, ROBOT.na, device=device))
            ep_len += 1
            done = te | to
            for name in tm.active_terms:
                hit = tm.get_term(name) & done
                if hit.any():
                    counts[name] += int(hit.sum())
                    frames.setdefault(name, []).extend(frame_before[hit].tolist())
            if done.any():
                lens += ep_len[done].tolist()
                ep_len[done] = 0
        print(f"assist level {assist}, action noise {action_noise_std}, reset noise {reset_noise}, DR {domain_randomization}: "
              f"{len(lens)} episodes ended, mean length {np.mean(lens) if lens else float('nan'):.1f} steps")
        for name, c in counts.most_common():
            f = np.array(frames[name])
            print(f"  {name:16s} {c:6d}  clip frame median {np.median(f):.0f}  quartiles {np.percentile(f, 25):.0f}-{np.percentile(f, 75):.0f}")
    finally:
        grip_assist.set_level(0.0)
        env.close()


if __name__ == "__main__":
    tyro.cli(main)

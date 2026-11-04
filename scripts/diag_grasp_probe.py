"""Grasp feasibility without a policy: replay the reference (zero residual) with the flex joints of each holding
hand closed by a fixed extra angle, one env per angle. Reports how long each env survives and how far the props lift.

  SM_ROBOT=capsule_hands_props uv run python scripts/diag_grasp_probe.py --closings 0 0.2 0.4 0.6 0.9 1.2
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.mdp.rewards import grip_pressure_agreement_sim
from supermimic.robots import ROBOT


@torch.no_grad()
def main(closings: list[float], clip: str = "sweep_task023", npz_dir: Path = Path("data/npz/capsule_hands"), anticipate: int = 10,
         device: str = "cuda:0") -> None:
    files = [str(f) for f in discover_npz(npz_dir, names=[clip])]
    n = len(closings)
    args = SuperMimicEnvCfgArgs(
        npz_files=files, num_envs=n, episode_length_s=1.0e5, action_noise_std=0.0, obs_noise=False, reset_noise=False,
        domain_randomization=False, sampling_mode="start", record_transitions=False, seed=0,
    )
    env = make_env(args, device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        term = env.action_manager.get_term("joint_pos")
        names = term.target_names
        scale = torch.tensor(ROBOT.action_scale_vector(), device=device)
        flex = torch.stack([torch.tensor([j.startswith(s) and j.endswith("_flex") for j in names], device=device) for s, _ in ROBOT.grip])  # (G, A)
        # Holding per frame, switched on `anticipate` frames early so the fingers close before the lift.
        hold = lib.grip_hold  # (N, G)
        early = hold.clone()
        for d in range(1, anticipate + 1):
            early[:-d] |= hold[d:]
        c = torch.tensor(closings, device=device)
        cmd.forced_frame[:] = lib.clip_start[torch.zeros(n, dtype=torch.long, device=device)]
        env.reset()
        alive = torch.ones(n, dtype=torch.bool, device=device)
        steps = torch.zeros(n, dtype=torch.long, device=device)
        z0 = cmd.sim_prop_pos_w[..., 2].clone()
        lift = torch.zeros(n, ROBOT.num_props, device=device)
        err = torch.zeros(n, ROBOT.num_props, device=device)
        agree, nhold = torch.zeros(n, len(ROBOT.grip), device=device), torch.zeros(n, len(ROBOT.grip), device=device)
        while alive.any():
            h = early[cmd.frame].float()  # (n, G)
            a = (h[:, :, None] * flex[None].float()).sum(1) * c[:, None] / scale
            _, _, te, to, _ = env.step(a)
            done = (te | to) & alive
            alive &= ~done
            steps += alive.long()
            live = alive[:, None].float()
            lift = torch.maximum(lift, (cmd.sim_prop_pos_w[..., 2] - z0) * live)
            err += (cmd.sim_prop_pos_w - cmd.prop_pos_w).norm(dim=-1) * live
            hn = hold[cmd.frame].float() * live
            agree += grip_pressure_agreement_sim(env, cmd) * hn
            nhold += hn
        L = lib.clips[0].length
        for i in range(n):
            props = ", ".join(f"{p} lift {lift[i, k] * 100:4.1f} cm err {err[i, k] / max(int(steps[i]), 1) * 100:4.1f} cm" for k, p in enumerate(ROBOT.props))
            gl = ", ".join(f"{s} agree {agree[i, g] / nhold[i, g].clamp(min=1):.2f}" for g, (s, _) in enumerate(ROBOT.grip))
            print(f"closing {closings[i]:5.2f} rad: steps {int(steps[i]):4d}/{L} | {props} | {gl}")
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)

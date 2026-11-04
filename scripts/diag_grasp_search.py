"""Search a hand's grasp offsets: cross-entropy search over per-joint-group set-point offsets (finger flex per
segment, finger spread, thumb flex / spread per segment) applied while the glove holds, on top of the reference
(zero residual), one candidate per env. The other gripping hands get a uniform flex closing of ``other_level`` rad.
Fitness: steps survived (to the prop-drop termination or ``horizon``) minus the held prop's mean position error.

  SM_ROBOT=capsule_hands_props uv run python scripts/diag_grasp_search.py --side L
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp import grip_assist
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.robots import ROBOT

GROUPS = {
    "flex1": r"{S}_(index|middle|ring|pinky)1_flex", "flex2": r"{S}_(index|middle|ring|pinky)2_flex",
    "flex3": r"{S}_(index|middle|ring|pinky)3_flex", "spread": r"{S}_(index|middle|ring|pinky)1_spread",
    "thumb1_flex": r"{S}_thumb1_flex", "thumb1_spread": r"{S}_thumb1_spread", "thumb2_flex": r"{S}_thumb2_flex",
    "thumb2_spread": r"{S}_thumb2_spread", "thumb3_flex": r"{S}_thumb3_flex",
}


@torch.no_grad()
def main(side: str = "L", clip: str = "sweep_task023", npz_dir: Path = Path("data/npz/capsule_hands"), pop: int = 512, elites: int = 32,
         generations: int = 6, horizon: int = 600, other_level: float = 0.6, init_flex: float = 0.6, init_std: float = 0.4,
         robust: bool = False, from_robot: bool = False, device: str = "cuda:0", seed: int = 0) -> None:
    """``robust``: score under the training perturbations (action / reset noise, domain randomisation).
    ``from_robot``: start from ``ROBOT.grip_assist`` and give the other hands their ``ROBOT.grip_assist`` grasp."""
    files = [str(f) for f in discover_npz(npz_dir, names=[clip])]
    env = make_env(SuperMimicEnvCfgArgs(npz_files=files, num_envs=pop, episode_length_s=1.0e5, action_noise_std=0.1 if robust else 0.0, obs_noise=False,
                                        reset_noise=robust, domain_randomization=robust, sampling_mode="start", record_transitions=False), device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        names = env.action_manager.get_term("joint_pos").target_names
        scale = torch.tensor(ROBOT.action_scale_vector(), device=device)
        G = torch.stack([torch.tensor([re.fullmatch(p.format(S=side), j) is not None for j in names], device=device) for p in GROUPS.values()]).float()
        g_side = [s for s, _ in ROBOT.grip].index(side)
        prop = ROBOT.props.index(dict(ROBOT.grip)[side])
        if from_robot:
            other = grip_assist.offsets(names, device)
            current = dict(ROBOT.grip_assist)
            mean = torch.tensor([current[p.format(S=side)] for p in GROUPS.values()], device=device)
        else:
            other = other_level * torch.stack([torch.tensor([j.startswith(f"{s}_") and j.endswith("_flex") for j in names], device=device) for s, _ in ROBOT.grip]).float()
            mean = torch.tensor([init_flex if k.endswith("flex") or k.startswith("flex") else 0.0 for k in GROUPS], device=device)
        other[g_side] = 0
        hold = lib.grip_hold_early
        gen = torch.Generator(device=device).manual_seed(seed)
        std = torch.full_like(mean, init_std)
        for it in range(generations):
            x = mean + std * torch.randn(pop, len(GROUPS), device=device, generator=gen)
            x[0] = mean
            off = x @ G  # (pop, A) rad
            cmd.forced_frame[:] = lib.clip_start[0]
            env.reset()
            alive = torch.ones(pop, dtype=torch.bool, device=device)
            steps = torch.zeros(pop, device=device)
            err = torch.zeros(pop, device=device)
            lift = torch.zeros(pop, device=device)
            z0 = cmd.sim_prop_pos_w[:, prop, 2].clone()
            for _ in range(horizon):
                h = hold[cmd.frame].float()
                a = (h[:, g_side, None] * off + h @ other) / scale
                _, _, te, to, _ = env.step(a)
                alive &= ~(te | to)
                steps += alive.float()
                err += (cmd.sim_prop_pos_w[:, prop] - cmd.prop_pos_w[:, prop]).norm(dim=-1) * alive
                lift = torch.maximum(lift, (cmd.sim_prop_pos_w[:, prop, 2] - z0) * alive)
                if not alive.any():
                    break
            mean_err = err / steps.clamp(min=1)
            fit = steps - 1000.0 * mean_err
            top = fit.topk(elites).indices
            b = int(top[0])
            print(f"gen {it}: mean-candidate steps {int(steps[0])} err {mean_err[0] * 100:.1f} cm | best steps {int(steps[b])} err {mean_err[b] * 100:.1f} cm "
                  f"lift {lift[b] * 100:.1f} cm | elite steps {steps[top].mean():.0f}", flush=True)
            mean, std = x[top].mean(0), x[top].std(0).clamp(min=0.05)
        print("best offsets (rad):", {k: round(float(v), 2) for k, v in zip(GROUPS, x[b])})
        print("elite mean (rad):  ", {k: round(float(v), 2) for k, v in zip(GROUPS, mean)})
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)

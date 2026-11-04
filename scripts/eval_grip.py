"""Grip robustness: play a clip from its first frame in many envs at once with the deterministic policy (no grip
assist) under chosen perturbations, and report how many finish the clip, how well the props are carried and, for each
gripping hand, the glove pressure agreement (supermimic.kinematics.glove.pressure_agreement) averaged over the frames
where its glove holds.

  SM_ROBOT=capsule_hands_props uv run python scripts/eval_grip.py --ckpt logs/capsule_grip_s3/ckpt_0007000.pt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.mdp.rewards import grip_pressure_agreement_sim
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.robots import ROBOT


@torch.no_grad()
def main(ckpt: Path, clip: str = "sweep_task023", npz_dir: Path = Path("data/npz/capsule_hands"), num_envs: int = 128,
         action_noise_std: float = 0.0, reset_noise: bool = True, domain_randomization: bool = True, obs_noise: bool = True,
         device: str = "cuda:0", seed: int = 1) -> None:
    files = [str(f) for f in discover_npz(npz_dir, names=[clip])]
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    env = make_env(SuperMimicEnvCfgArgs(npz_files=files, num_envs=num_envs, future_frames=cfg["env"]["future_frames"], episode_length_s=1.0e5,
                                        action_noise_std=action_noise_std, obs_noise=obs_noise, reset_noise=reset_noise,
                                        domain_randomization=domain_randomization, sampling_mode="start", record_transitions=False, seed=seed), device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        L = lib.clips[0].length
        cmd.forced_frame[:] = lib.clip_start[0]
        obs, _ = env.reset()
        alive = torch.ones(num_envs, dtype=torch.bool, device=device)
        steps = torch.zeros(num_envs, device=device)
        err = torch.zeros(num_envs, ROBOT.num_props, device=device)
        z0 = cmd.sim_prop_pos_w[..., 2].clone()
        lift = torch.zeros(num_envs, ROBOT.num_props, device=device)
        G = len(ROBOT.grip)
        agree, hold_n = torch.zeros(G, device=device), torch.zeros(G, device=device)
        while alive.any():
            obs, _, te, to, _ = env.step(policy(obs["policy"], obs["goal"]))
            done = (te | to) & alive
            dropped = done & te
            alive &= ~done
            steps += alive.float() + (done & ~dropped).float()
            live = alive[:, None].float()
            err += (cmd.sim_prop_pos_w - cmd.prop_pos_w).norm(dim=-1) * live
            lift = torch.maximum(lift, (cmd.sim_prop_pos_w[..., 2] - z0) * live)
            if G:
                h = lib.grip_hold[cmd.frame].float() * live  # (n, G)
                agree += (grip_pressure_agreement_sim(env, cmd) * h).sum(0)
                hold_n += h.sum(0)
            if steps.max() > L + 5:
                break
        full = steps >= L - 2
        print(f"{ckpt.name}: action noise {action_noise_std}, obs noise {obs_noise}, reset noise {reset_noise}, DR {domain_randomization}")
        print(f"  finished the clip: {int(full.sum())}/{num_envs} ({full.float().mean():.0%}); steps median {steps.median():.0f}, quartiles "
              f"{np.percentile(steps.cpu(), 25):.0f}-{np.percentile(steps.cpu(), 75):.0f}")
        for k, p in enumerate(ROBOT.props):
            e = (err[:, k] / steps.clamp(min=1))[full]
            print(f"  {p}: lift median {lift[:, k].median() * 100:.1f} cm; mean error over finished episodes {e.mean() * 100 if full.any() else float('nan'):.1f} cm")
        a = (agree / hold_n.clamp(min=1)).tolist()
        for g, (side, prop) in enumerate(ROBOT.grip):
            print(f"  {side} hand ({prop}): glove pressure agreement while holding {a[g]:.3f}")
        print("SUMMARY " + " ".join([f"ckpt={ckpt.stem}", f"finished={full.float().mean():.4f}"]
                                    + [f"lift_{p}={lift[:, k].median() * 100:.2f}" for k, p in enumerate(ROBOT.props)]
                                    + [f"agree_{side}={a[g]:.4f}" for g, (side, _) in enumerate(ROBOT.grip)]))
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)

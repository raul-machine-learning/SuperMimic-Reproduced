"""Record deterministic policy rollouts (robot qpos + reference qpos per control step) for rendering.

  uv run python scripts/media/record_rollout.py --ckpt logs/e8/ckpt_0100000.pt --npz-dir data/npz/bones_seed \
      --clips cartwheelin_003__A486_M jump_ff_360_R_004__A407 --out media/rollouts.npz

Same protocol as the Appendix F benchmark (scripts/eval_benchmark.py): every clip starts at its first frame,
no observation/action noise, no reset noise, no domain randomization. Each clip is played once; recording
stops at the clip end or at the termination. Render the result with scripts/media/render_rollout.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp import grip_assist
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.mdp.rewards import TrackingRewardCfg, sim_glove, sim_taxel_forces
from supermimic.envs.state_io import NQ, StateIO
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.robots import ROBOT


@torch.no_grad()
def main(ckpt: Path, clips: list[str], npz_dir: Path = Path("data/npz/bones_seed"), out: Path = Path("media/rollouts.npz"), device: str = "cuda:0",
         grip_assist_level: float = 0.0) -> None:
    """``grip_assist_level``: fraction of the grasp offsets added while the glove holds (supermimic.envs.mdp.grip_assist); 0 = the policy alone."""
    grip_assist.set_level(grip_assist_level)
    files = [str(f) for f in discover_npz(npz_dir, names=clips)]
    missing = set(clips) - {Path(f).stem for f in files}
    assert not missing, f"clips not found in {npz_dir}: {sorted(missing)}"
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    n = len(files)
    args = SuperMimicEnvCfgArgs(
        npz_files=files, num_envs=n, future_frames=cfg["env"]["future_frames"], episode_length_s=1.0e5,
        action_noise_std=0.0, obs_noise=False, reset_noise=False, domain_randomization=False, sampling_mode="start",
        record_transitions=False, seed=0, anchor_obs=cfg["env"].get("anchor_obs", False),
    )
    env = make_env(args, device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        io = StateIO(env)
        env_clip = torch.arange(n, device=device)
        cmd.forced_frame[:] = lib.clip_start[env_clip]
        obs, _ = env.reset()
        cmd.forced_frame[:] = lib.clip_start[env_clip]  # keep the clip if an env is reset again
        max_r = TrackingRewardCfg().max_step_reward  # paper reward upper bound per step (G1: dt * 5)
        rob, ref, props, grip, taxel_force = [], [], [], [], []
        alive = torch.ones(n, dtype=torch.bool, device=device)
        rew_sum = torch.zeros(n, device=device)
        steps = torch.zeros(n, dtype=torch.long, device=device)

        def snapshot() -> None:
            s = io.read()
            rob.append(s[:, :NQ].cpu().numpy())
            if ROBOT.props:
                props.append(s[:, ROBOT.robot_state_dim :].reshape(n, ROBOT.num_props, 13)[..., :7].cpu().numpy())
                g = [(cmd.sim_prop_pos_w - cmd.prop_pos_w).norm(dim=-1)]  # (n, P) prop position error
                if ROBOT.grip:
                    f = sim_taxel_forces(env, cmd)
                    g += [sim_glove(cmd).force_agreement(f, cmd.taxel_force), cmd.taxel_pressed.amax(-1)]  # (n, G) agreement, holding in the reference
                    taxel_force.append(torch.stack([f, cmd.taxel_force], 2).cpu().numpy().astype(np.float16))  # (n, G, 2, 129) sim, glove
                grip.append(torch.cat(g, -1).cpu().numpy())
            ref.append(lib.ref_qpos_qvel(cmd.frame)[0].cpu().numpy())

        snapshot()
        while alive.any():
            act = policy(obs["policy"], obs["goal"])
            obs, rew, term, tout, _ = env.step(act)
            rew_sum += rew * alive
            steps += alive.long()
            done = (term | tout) & alive
            alive &= ~done
            if alive.any():
                snapshot()
            if steps.max() > 20000:
                break
        names = [lib.clips[i].name for i in range(n)]
        lengths = steps.cpu().numpy()  # recorded frames = steps (the terminal/reset state is not recorded)
        T = len(rob)
        robot_qpos, ref_qpos = np.stack(rob, 1), np.stack(ref, 1)
        for i in range(n):  # freeze frames after a clip's end so every row has the same length
            robot_qpos[i, lengths[i]:] = robot_qpos[i, lengths[i] - 1]
            ref_qpos[i, lengths[i]:] = ref_qpos[i, lengths[i] - 1]
        out.parent.mkdir(parents=True, exist_ok=True)
        norm_return = (rew_sum / (max_r * steps.clamp(min=1))).cpu().numpy()
        extra = {}
        if ROBOT.props:
            from supermimic.envs.props import PROP_SOURCES

            prop_qpos = np.stack(props, 1)
            for i in range(n):
                prop_qpos[i, lengths[i]:] = prop_qpos[i, lengths[i] - 1]
            extra = dict(prop_qpos=prop_qpos[:, :T], prop_names=np.array(ROBOT.props), prop_source_names=np.array([PROP_SOURCES[p] for p in ROBOT.props]))
            g = np.stack(grip, 1)[:, :T]
            P, G = ROBOT.num_props, len(ROBOT.grip)
            extra["prop_err"] = g[..., :P]
            if G:
                extra["grip_force_agreement"], extra["ref_holding"] = g[..., P : P + G], g[..., P + G :] > 0.5
                extra["taxel_force"] = np.stack(taxel_force, 1)[:, :T]  # (n, T, G, 2, 129) N: simulated, glove
            for i in range(n):
                L = int(lengths[i])
                line = "  props: " + ", ".join(f"{p} error mean {g[i, :L, k].mean() * 100:.1f} cm max {g[i, :L, k].max() * 100:.1f} cm"
                                               f", lift {(prop_qpos[i, :L, k, 2].max() - prop_qpos[i, 0, k, 2]) * 100:.1f} cm" for k, p in enumerate(ROBOT.props))
                for h, (side, prop) in enumerate(ROBOT.grip):
                    hold = g[i, :L, P + G + h] > 0.5
                    tf = extra["taxel_force"][i, :L, h].astype(np.float32).sum(-1)  # (L, 2)
                    line += (f" | {side} force agreement holding {g[i, :L, P + h][hold].mean():.3f} free {g[i, :L, P + h][~hold].mean():.3f}, "
                             f"grip {np.median(tf[hold, 0]):.1f} N (glove {np.median(tf[hold, 1]):.1f} N)")
                print(line)
        np.savez_compressed(out, names=np.array(names), robot_qpos=robot_qpos[:, :T], ref_qpos=ref_qpos[:, :T], lengths=lengths, norm_return=norm_return, **extra)
        for i, nm in enumerate(names):
            print(f"{nm:48s} steps {int(lengths[i]):5d}/{lib.clips[i].length}  return {float(norm_return[i]):.3f}")
        print("saved", out)
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)

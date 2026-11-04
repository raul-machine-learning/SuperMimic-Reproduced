"""Simulated grip forces against the glove: play the clip from its first frame (a policy checkpoint without assist,
or the reference with the grasp assist at ``--assist``) and read, on every body of each gripping hand, the contact
normal force against the prop that hand holds. Compares them with the glove's load above its rest baseline, summed
over the taxels on each body, and saves both per frame.

  SM_ROBOT=capsule_hands_props uv run python scripts/diag_grip_forces.py --ckpt logs/capsule_grip_s3/ckpt_0007000.pt
  SM_ROBOT=capsule_hands_props uv run python scripts/diag_grip_forces.py --assist 1.0
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs
from supermimic.envs.mdp import grip_assist
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.tracking_env_cfg import make_env_cfg
from supermimic.robots import ROBOT

HANDS_SRC = Path("data/hands/sweep_task023_hands_registered_closed.npz")


def glove_body_load(times: np.ndarray) -> tuple[list[list[str]], np.ndarray]:
    """Per gripping hand, the bodies carrying taxels and (T, G, B) the glove load summed over each body's taxels."""
    from supermimic.data.glove_taxels import fitted_adjust, taxel_layout
    from supermimic.data.sweep_mocap import taxel_load
    from supermimic.kinematics.glove import HANDS_XML

    m = mujoco.MjModel.from_xml_path(str(HANDS_XML))
    z = np.load(HANDS_SRC)
    names, loads = [], []
    for side, _ in ROBOT.grip:
        lay = taxel_layout(m, side, fitted_adjust())
        load = taxel_load(z[f"pressure_{'left' if side == 'L' else 'right'}"], z["pressure_times"], times)
        bodies = sorted(set(int(b) for b in lay.body))
        names.append([m.body(b).name for b in bodies])
        loads.append(np.stack([load[:, lay.body == b].sum(1) for b in bodies], 1))
    nb = max(len(n) for n in names)
    out = np.zeros((len(times), len(names), nb))
    for g, ld in enumerate(loads):
        out[:, g, : ld.shape[1]] = ld
    return names, out


@torch.no_grad()
def main(ckpt: Path | None = None, assist: float = 0.0, clip: str = "sweep_task023", npz_dir: Path = Path("data/npz/capsule_hands"),
         num_envs: int = 32, perturb: bool = True, device: str = "cuda:0", out: Path = Path("logs/grip_forces.npz"), seed: int = 1) -> None:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg

    files = [str(f) for f in discover_npz(npz_dir, names=[clip])]
    policy = None
    future = 10
    if ckpt is not None:
        from supermimic.eval.benchmark import load_policy_from_checkpoint

        policy, cfg = load_policy_from_checkpoint(ckpt, device)
        future = cfg["env"]["future_frames"]
    times = np.load(files[0])["times"].astype(np.float64)
    body_names, glove = glove_body_load(times)

    args = SuperMimicEnvCfgArgs(npz_files=files, num_envs=num_envs, future_frames=future, episode_length_s=1.0e5, action_noise_std=0.0,
                                obs_noise=perturb, reset_noise=perturb, domain_randomization=perturb, sampling_mode="start",
                                record_transitions=False, seed=seed)
    env_cfg = make_env_cfg(args)
    env_cfg.scene.sensors = tuple(
        ContactSensorCfg(name=f"grip_{side}", primary=ContactMatch(mode="body", pattern=tuple(body_names[g]), entity="robot"),
                         secondary=ContactMatch(mode="body", pattern=prop, entity=prop), fields=("found", "force"), reduce="maxforce", num_slots=4)
        for g, (side, prop) in enumerate(ROBOT.grip)
    )
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    env = ManagerBasedRlEnv(env_cfg, device=device)
    grip_assist.set_level(assist)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        L = lib.clips[0].length
        cmd.forced_frame[:] = lib.clip_start[0]
        obs, _ = env.reset()
        sensors = [env.scene[f"grip_{side}"] for side, _ in ROBOT.grip]
        nb = glove.shape[2]
        force = np.full((L, num_envs, len(ROBOT.grip), nb), np.nan)
        alive = torch.ones(num_envs, dtype=torch.bool, device=device)
        zero = torch.zeros(num_envs, ROBOT.na, device=device)
        for t in range(L - 1):
            a = policy(obs["policy"], obs["goal"]) if policy is not None else zero
            obs, _, te, to, _ = env.step(a)
            alive &= ~(te | to)
            f = cmd.frame - lib.clip_start[0]
            for g, s in enumerate(sensors):
                d = s.data
                fn = (d.force[..., 0].clamp(min=0.0) * (d.found > 0)).view(num_envs, -1, 4).sum(-1)  # (n, B) normal force
                live = alive.cpu().numpy()
                fr = f.cpu().numpy()
                force[fr[live], np.nonzero(live)[0], g, : fn.shape[1]] = fn.cpu().numpy()[live]
            if not alive.any():
                break
        n_alive = int(alive.sum())
        print(f"{'ckpt ' + ckpt.name if ckpt else f'reference + assist {assist:.0%}'}: {n_alive}/{num_envs} finished the clip")
        hold = lib.grip_hold.cpu().numpy()[: L]
        for g, (side, prop) in enumerate(ROBOT.grip):
            h = hold[:, g]
            fs = np.nanmean(force[:, :, g], axis=1)  # (L, B) mean over live envs
            tot_sim = np.nansum(fs[h], -1)
            tot_glove = glove[h, g].sum(-1)
            k = np.median(tot_sim) / max(np.median(tot_glove), 1e-6)
            print(f"\n{side} hand / {prop}: {h.sum()} holding frames; total normal force median {np.median(tot_sim):.2f} N "
                  f"(quartiles {np.percentile(tot_sim, 25):.2f}-{np.percentile(tot_sim, 75):.2f}); glove load median {np.median(tot_glove):.0f} raw "
                  f"-> {k:.3f} N per raw unit")
            corr = np.corrcoef(tot_sim, tot_glove)[0, 1] if tot_sim.std() > 0 else float("nan")
            print(f"  correlation over time of total force with total glove load: {corr:.2f}")
            share_sim = np.nanmean(fs[h], 0) / max(np.nanmean(tot_sim), 1e-9)
            share_glove = glove[h, g].mean(0) / max(tot_glove.mean(), 1e-9)
            print(f"  {'body':12s} {'sim N':>7s} {'sim %':>6s} {'glove %':>8s}")
            for b, name in enumerate(body_names[g]):
                print(f"  {name:12s} {np.nanmean(fs[h, b]):7.2f} {share_sim[b] * 100:6.1f} {share_glove[b] * 100:8.1f}")
            overlap = np.minimum(share_sim[: len(body_names[g])], share_glove[: len(body_names[g])]).sum()
            print(f"  distribution overlap (sum of min shares): {overlap:.2f}")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, force=force, glove=glove, hold=hold, body_names=np.array(body_names, dtype=object), times=times[:L])
        print(f"\nwrote {out}")
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)

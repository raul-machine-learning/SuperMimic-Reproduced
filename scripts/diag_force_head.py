"""How well a world model's grip force head predicts the simulator's contact forces: play a policy (no assist) under
the training perturbations, and on every simulated state compare the head's forces with the contact sensors', and
the pressure agreement with the glove computed from each.

  SM_ROBOT=capsule_hands_props uv run python scripts/diag_force_head.py --ckpt logs/capsule_grip_force_s1/ckpt_0001000.pt \
      --policy-ckpt logs/capsule_grip_s3/ckpt_0007000.pt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.robots import HAND_CLIPS, ROBOT


@torch.no_grad()
def main(ckpt: Path, policy_ckpt: Path | None = None, npz_dir: Path = HAND_CLIPS, num_envs: int = 256, steps: int = 400,
         device: str = "cuda:0", seed: int = 2, fit_steps: int = 0) -> None:
    """``fit_steps`` > 0: also fit a fresh force head on three quarters of the envs' true states and report it on the
    rest (how accurate the head can get without the world model's prediction drift)."""
    from supermimic.data.motion_convert import load_robot_model
    from supermimic.envs.mdp.grip_force import body_forces
    from supermimic.envs.state_io import StateIO
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.world_model.heads import AnalyticHeads
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    policy, cfg = load_policy_from_checkpoint(policy_ckpt or ckpt, device)
    env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(f) for f in discover_npz(npz_dir)], num_envs=num_envs, future_frames=cfg["env"]["future_frames"],
                                        record_transitions=False, seed=seed), device)
    try:
        robot = env.scene["robot"]
        heads = AnalyticHeads(G1Kinematics(load_robot_model(), device=device), TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY,
                              robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone())
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        from supermimic.world_model.state_spec import StateSpecCfg

        tup = lambda c: {k: tuple(v) if isinstance(v, list) else v for k, v in c.items()}  # noqa: E731
        wm_raw = ck["config"]["world_model"]
        wm_cfg = WorldModelCfg(**{**tup({k: v for k, v in wm_raw.items() if k != "state_spec"}), "state_spec": StateSpecCfg(**tup(wm_raw["state_spec"]))})
        wm = WorldModel(wm_cfg, heads, device)
        wm.load_state_dict(ck["wm"])
        io = StateIO(env)
        term = env.action_manager.get_term("joint_pos")
        cmd = env.command_manager.get_term("motion")
        obs, _ = env.reset()
        pred_f, true_f, pred_a, true_a, hold, data = [], [], [], [], [], []
        for _ in range(steps):
            obs, *_ = env.step(policy(obs["policy"], obs["goal"]))
            s = io.read()
            d, fp = wm.grip_forces(s, term.executed_abs)
            ft = body_forces(env)
            ref = cmd.taxel_load
            pred_f.append(fp.cpu())
            true_f.append(ft.cpu())
            pred_a.append(heads.glove.pressure_agreement(heads.glove.taxel_forces(d, fp), ref).cpu())
            true_a.append(heads.glove.pressure_agreement(heads.glove.taxel_forces(d, ft), ref).cpu())
            hold.append(cmd.library.grip_hold[cmd.frame].cpu())
            if fit_steps:
                data.append((s.clone(), term.executed_abs.clone(), ft.clone(), d.clone(), ref.clone()))
        pf, tf, pa, ta, h = (torch.cat(x).numpy() for x in (pred_f, true_f, pred_a, true_a, hold))
        print(f"{ckpt}: {len(h)} states")
        if fit_steps:
            report_fresh_fit(wm, data, hold, fit_steps, num_envs)
        for g, (side, prop) in enumerate(ROBOT.grip):
            m = h[:, g]
            tot_p, tot_t = pf[m, g].sum(-1), tf[m, g].sum(-1)
            r2 = 1 - ((tot_p - tot_t) ** 2).mean() / tot_t.var()
            mae, mae0 = np.abs(pf[m, g] - tf[m, g]).mean(), np.abs(tf[m, g]).mean()
            print(f"  {side} / {prop}, holding states {m.sum()}: total force true median {np.median(tot_t):.1f} N, predicted {np.median(tot_p):.1f} N, "
                  f"R2 {r2:.2f}; per-body abs error {mae:.2f} N (predicting 0: {mae0:.2f} N)")
            c = np.corrcoef(pa[m, g], ta[m, g])[0, 1]
            print(f"     agreement with the glove: true {ta[m, g].mean():.3f}, from predicted forces {pa[m, g].mean():.3f}, correlation {c:.2f}; "
                  f"free hand true {ta[~m, g].mean():.3f} predicted {pa[~m, g].mean():.3f}")
    finally:
        env.close()


def report_fresh_fit(wm, data, hold, fit_steps: int, num_envs: int) -> None:
    import copy

    from supermimic.world_model.world_model import force_loss

    head = copy.deepcopy(wm.force_head)
    for m in head.modules():
        if isinstance(m, torch.nn.Linear):
            m.reset_parameters()
    torch.nn.init.constant_(head.mlp.net[-1].bias, -3.0)
    s, a, ft, d, ref = (torch.stack(x) for x in zip(*data))  # (T, n, ·)
    h = torch.stack(hold).to(s.device)
    feat = wm.spec.canonical(s)
    q = ROBOT.split_state(s)[2][..., wm.act_idx]
    push = a - (q - wm.act_q0) / wm.act_scale
    ntr = num_envs * 3 // 4
    flat = lambda x, sl: x[:, sl].flatten(0, 1)  # noqa: E731
    tr = [flat(x, slice(0, ntr)) for x in (feat, push, d, ft)]
    te = [flat(x, slice(ntr, None)) for x in (feat, push, d, ft, ref, h)]
    with torch.enable_grad():
        opt = torch.optim.Adam(head.parameters(), lr=3e-4)
        for i in range(fit_steps):
            idx = torch.randint(0, tr[0].shape[0], (4096,), device=s.device)
            loss = force_loss(head(tr[0][idx], tr[1][idx], tr[2][idx]), tr[3][idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    glove = wm.heads.glove
    pred = torch.cat([head(te[0][i : i + 8192], te[1][i : i + 8192], te[2][i : i + 8192]) for i in range(0, te[0].shape[0], 8192)])
    pa = glove.pressure_agreement(glove.taxel_forces(te[2], pred), te[4]).cpu().numpy()
    ta = glove.pressure_agreement(glove.taxel_forces(te[2], te[3]), te[4]).cpu().numpy()
    pred, true, hh = pred.cpu().numpy(), te[3].cpu().numpy(), te[5].cpu().numpy()
    print(f"  fresh head fit on true states ({fit_steps} steps), held-out envs:")
    for g, (side, _) in enumerate(ROBOT.grip):
        m = hh[:, g]
        tot_p, tot_t = pred[m, g].sum(-1), true[m, g].sum(-1)
        r2 = 1 - ((tot_p - tot_t) ** 2).mean() / tot_t.var()
        print(f"    {side}: total force R2 {r2:.2f}, per-body abs error {np.abs(pred[m, g] - true[m, g]).mean():.2f} N; agreement correlation "
              f"{np.corrcoef(pa[m, g], ta[m, g])[0, 1]:.2f}; free hand true {ta[~m, g].mean():.3f} predicted {pa[~m, g].mean():.3f}")


if __name__ == "__main__":
    tyro.cli(main)

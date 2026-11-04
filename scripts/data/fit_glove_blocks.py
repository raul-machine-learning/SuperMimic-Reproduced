"""Fit the center and scale of each glove taxel block to the glove's pressure at the mocap pose, by gradient descent.

  SM_ROBOT=capsule_hands_props uv run python scripts/data/fit_glove_blocks.py          (SM_HAND_SHAPE=box: box hands)

Blocks (``supermimic.data.glove_taxels.block_bounds``): every fingertip pad (thumb 4 x 3, fingers 3 x 3) and the palm
(five rows) is an array with a center and a scale of its spacing; the single taxels (middle and proximal phalanges,
thumb base) have a center. The parameters are anatomical (toward the tip, the finger bases, the index side), so the
same values place mirror-image layouts on the two hands: with ``mirror`` (default) one set is fitted to both hands,
the left glove being the mirror of the right; ``--no-mirror`` fits one set per hand.

Data: every frame where a glove holds, with the hands and props written at the rig clip's mocap pose and one physics
step (as scripts/diag_pressure_blocks.py --mocap-pose-frames): the contact normal force of each hand segment on its
prop, the hand and prop poses, the glove's pressure. The segment forces do not depend on where the taxels sit, so the
fit runs in torch alone: block parameters -> taxel positions (``glove_taxels.taxel_layout_torch``) -> signed distance
to the prop -> each segment's force spread over its taxels -> the per-hand shares of simulated force and of glove
pressure with their "nothing" share, as in ``GloveModel.pressure_agreement``. Loss: the squared error of the shares
summed over the 130 bins, mean over frames, plus a small pull toward the neutral layout. Every 4th half-second of the
holds is held out. Reports the loss and the overlap score (``pressure_agreement``) on the fit and held-out frames,
writes ``glove_taxels.BLOCK_FIT_FILE`` (read by the glove reward), checks the reward's ``GloveModel`` against it
and writes a picture of the fitted sensor positions on the hands (scripts/media/render_glove_blocks.py) to ``image``
(default media/glove_block_fit_<shape>.png).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import torch
import torch.nn.functional as F
import tyro

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))


def collect(device: str) -> dict:
    """Mocap-pose physics on every frame where a glove holds: poses, segment forces and glove pressure."""
    from sweep_data import NPZ, sweep_files

    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.grip_force import body_forces
    from supermimic.envs.mdp.rewards import sim_glove
    from supermimic.envs.state_io import StateIO
    from supermimic.robots import ROBOT

    frames = np.nonzero(np.load(NPZ)["taxel_load"].sum(-1).max(-1) > 0)[0]
    env = make_env(SuperMimicEnvCfgArgs(npz_files=sweep_files(1), num_envs=len(frames), obs_noise=False, reset_noise=False,
                                        domain_randomization=False, action_noise_std=0.0, sampling_mode="start", seed=0), device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        cmd.forced_frame[:] = lib.clip_start[0] + torch.as_tensor(frames, device=env.device)
        env.reset()
        StateIO(env).write(torch.arange(env.num_envs, device=env.device), lib._ref[cmd.frame].float())
        env.step(torch.zeros(env.num_envs, ROBOT.na, device=env.device))
        glove = sim_glove(cmd)
        return dict(
            body_pos=cmd.robot.data.body_link_pos_w.clone(), body_quat=cmd.robot.data.body_link_quat_w.clone(),
            prop_pos=cmd.sim_prop_pos_w.clone(), prop_quat=cmd.sim_prop_quat_w.clone(), body_force=body_forces(env).clone(),
            load=cmd.taxel_load.clone(), time=((cmd.frame - lib.clip_start[0]).float() / 50.0).cpu().numpy(),
            body_names=list(cmd.robot.body_names), glove=glove,
        )
    finally:
        env.close()


def shares(x: torch.Tensor, floor: float) -> torch.Tensor:
    """(..., 130): each taxel's share of the total plus the floor, and the "nothing" share."""
    p = x / (x.sum(-1, keepdim=True) + floor)
    return torch.cat([p, 1.0 - p.sum(-1, keepdim=True)], -1)


def main(steps: int = 300, lr: float = 0.05, prior: float = 2e-3, mirror: bool = True, device: str = "cuda:0", write: bool = True,
         image: Path | None = None) -> None:
    from diag_pressure_blocks import BLOCKS

    from supermimic.data.glove_taxels import BLOCK_FIT_FILE, block_bounds, neutral_blocks, taxel_layout_torch
    from supermimic.kinematics.glove import GLOVE_PRESSURE_FLOOR, NEAR, OFF_TAXEL_SHARE, SIM_PRESSURE_FLOOR, SOFT, GloveModel
    from supermimic.kinematics.rotations import quat_apply, quat_apply_inverse
    from supermimic.robots import HAND_SHAPE, HANDS_XML, ROBOT

    D = collect(device)
    glove = D["glove"]
    model = mujoco.MjModel.from_xml_path(str(HANDS_XML))
    index = {n: i for i, n in enumerate(D["body_names"])}
    bounds, neutral = block_bounds(), neutral_blocks()
    keys = list(bounds)
    lo = torch.tensor([b[0] for k in keys for b in bounds[k]], dtype=torch.float64)
    hi = torch.tensor([b[1] for k in keys for b in bounds[k]], dtype=torch.float64)
    mid = torch.tensor([v for k in keys for v in neutral[k]], dtype=torch.float64)
    z0 = torch.atanh(2.0 * (mid - lo) / (hi - lo) - 1.0)
    sizes = [len(bounds[k]) for k in keys]
    val = (np.floor(D["time"] / 0.5) % 4 == 3)
    hold = D["load"].sum(-1) > 0  # (N, G)

    def values(z: torch.Tensor) -> torch.Tensor:
        return lo + (hi - lo) * (torch.tanh(z + z0) + 1.0) / 2.0

    def unpack(v: torch.Tensor) -> dict[str, torch.Tensor]:
        return dict(zip(keys, torch.split(v, sizes)))

    def forward(h: int, S: str, v: torch.Tensor, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(simulated, glove) shares (F, 130) of hand h on the frames ``rows``."""
        body, pos, _ = taxel_layout_torch(model, S, unpack(v))
        names = [model.body(int(b)).name for b in body]
        bidx = torch.tensor([index[n] for n in names], device=device)
        slot = torch.tensor([glove.force_body_names[h].index(n) for n in names], device=device)
        local = pos.float().to(device)
        bp, bq = D["body_pos"][rows][:, bidx], D["body_quat"][rows][:, bidx]
        x = bp + quat_apply(bq, local.expand_as(bp))
        k = glove.prop_index[h]
        pp, pq = D["prop_pos"][rows, k, None], D["prop_quat"][rows, k, None].expand(-1, 129, 4)
        u = 2.0 * (quat_apply_inverse(pq, x - pp) - glove.lo[h]) / glove.span[h] - 1.0
        d = F.grid_sample(glove.sdf[h], u.reshape(1, 1, 1, -1, 3), mode="bilinear", padding_mode="border", align_corners=True).reshape(len(rows), 129)
        onehot = F.one_hot(slot, glove.num_bodies).float()  # (129, B)
        w = torch.sigmoid((NEAR - d) / SOFT) + OFF_TAXEL_SHARE
        share = w / ((w @ onehot) @ onehot.T)
        force = share * (D["body_force"][rows, h] @ onehot.T)
        return shares(force, SIM_PRESSURE_FLOOR), shares(D["load"][rows, h], GLOVE_PRESSURE_FLOOR)

    def score(h: int, S: str, v: torch.Tensor, mask: np.ndarray) -> tuple[float, float]:
        rows = torch.as_tensor(np.nonzero(mask & hold[:, h].cpu().numpy())[0], device=device)
        with torch.no_grad():
            P, Q = forward(h, S, v, rows)
        return ((P - Q) ** 2).sum(-1).mean().item(), torch.minimum(P, Q).sum(-1).mean().item()

    sides = [s for s, _ in ROBOT.grip]
    shared = torch.zeros(len(lo), dtype=torch.float64, requires_grad=True)
    z = {S: shared if mirror else torch.zeros(len(lo), dtype=torch.float64, requires_grad=True) for S in sides}
    before = {S: {"fit": score(h, S, values(z[S].detach()), ~val), "held_out": score(h, S, values(z[S].detach()), val)} for h, S in enumerate(sides)}
    opt = torch.optim.Adam([shared] if mirror else list(z.values()), lr=lr)
    fit_rows = {h: torch.as_tensor(np.nonzero(~val & hold[:, h].cpu().numpy())[0], device=device) for h in range(len(sides))}
    for it in range(steps):
        opt.zero_grad()
        total = 0.0
        for h, S in enumerate(sides):
            v = values(z[S])
            P, Q = forward(h, S, v, fit_rows[h])
            loss = ((P - Q) ** 2).sum(-1).mean() + prior * (((v - mid) / (hi - lo)) ** 2).sum().to(P.device)
            loss.backward()
            total += loss.item()
        opt.step()
        if it % 50 == 0 or it == steps - 1:
            print(f"step {it:4d}: loss {total:.4f}")

    fitted = {S: {k: [round(float(x), 6) for x in t] for k, t in unpack(values(z[S].detach())).items()} for S in sides}
    after = {S: {"fit": score(h, S, values(z[S].detach()), ~val), "held_out": score(h, S, values(z[S].detach()), val)} for h, S in enumerate(sides)}
    print(f"\n{HAND_SHAPE} hands, {'one mirrored set for both hands' if mirror else 'one set per hand'}, {len(D['time'])} frames ({int((~val).sum())} fit, {int(val.sum())} held out); "
          "share MSE (lower is better) and overlap score (pressure_agreement, higher is better)")
    for h, S in enumerate(sides):
        for part in ("fit", "held_out"):
            (m0, o0), (m1, o1) = before[S][part], after[S][part]
            print(f"  {S} {part:8s}  MSE {m0:.4f} -> {m1:.4f}   overlap {o0:.3f} -> {o1:.3f}")

    print("\nblock parameters: center shifts (mm) along the segment / toward the finger bases, toward the thumb / index side; scale")
    for k in keys:
        cells = []
        for S in sides[:1] if mirror else sides:
            t = fitted[S][k]
            cells.append(f"{'both' if mirror else S} {t[0] * 1000:+5.1f} {t[1] * 1000:+5.1f}" + (f"  x{t[2]:.2f}" if len(t) > 2 else "       "))
        print(f"  {k:16s} " + "   ".join(cells))

    print("\nper block after the fit (all holding frames): glove share, simulated share, overlap")
    for h, S in enumerate(sides):
        rows = torch.as_tensor(np.nonzero(hold[:, h].cpu().numpy())[0], device=device)
        with torch.no_grad():
            P0, Q = forward(h, S, values(z[S].detach() * 0), rows)
            P1, _ = forward(h, S, values(z[S].detach()), rows)
        print(f"  {S} hand: {'block':12s} {'glove':>6s} {'before':>7s} {'after':>6s} {'overlap before':>15s} {'after':>6s}")
        for name, ix in BLOCKS.items():
            ix = list(ix)
            print(f"          {name:12s} {Q[:, ix].sum(-1).mean().item():6.3f} {P0[:, ix].sum(-1).mean().item():7.3f} {P1[:, ix].sum(-1).mean().item():6.3f} "
                  f"{torch.minimum(P0, Q)[:, ix].sum(-1).mean().item():15.3f} {torch.minimum(P1, Q)[:, ix].sum(-1).mean().item():6.3f}")

    if write:
        BLOCK_FIT_FILE.write_text(json.dumps(dict(blocks=fitted, mirror=mirror, loss="squared error of the pressure shares", before=before, after=after,
                                                  frames=int(len(D["time"])), held_out=int(val.sum())), indent=1))
        print(f"\nwrote {BLOCK_FIT_FILE}")
        g = GloveModel(lambda names: torch.tensor([index[n] for n in names], device=device), device)
        dist = g.distances(D["body_pos"], D["body_quat"], D["prop_pos"], D["prop_quat"])
        agree = g.pressure_agreement(g.taxel_forces(dist, D["body_force"]), D["load"])
        for h, S in enumerate(sides):
            m = hold[:, h]
            print(f"  reward's GloveModel with the fit: {S} overlap {agree[m, h].mean().item():.3f} on all holding frames")
        spec = importlib.util.spec_from_file_location("render_glove_blocks", ROOT / "scripts/media/render_glove_blocks.py")
        render = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(render)
        render.main(image or ROOT / "media" / f"glove_block_fit_{HAND_SHAPE}.png")


if __name__ == "__main__":
    tyro.cli(main)

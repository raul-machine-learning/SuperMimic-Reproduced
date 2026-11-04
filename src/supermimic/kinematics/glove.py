"""Glove pressure grip term of the capsule hands (torch, differentiable).

Each hand's 129 glove taxels sit at fixed points of its bodies (``supermimic.data.glove_taxels``, the layout fitted
to the glove's pattern). The simulated taxels measure pressure: the simulator reads the contact normal force between
each taxel-carrying body (``body_names``, the *force bodies*) and the held prop (``ROBOT.grip``), and
``taxel_forces`` spreads a body's force over its taxels by where the body meets the prop, estimated from each
taxel's signed distance ``d`` to the prop surface (``proximity``: ``sigmoid((NEAR - d) / SOFT)`` plus a small uniform
share, so a contact off the taxels still counts); ``d`` is read from the prop's signed-distance grid
(scripts/assets/build_prop_sdf.py) by trilinear interpolation.

The glove is uncalibrated, so the comparison is relative: ``pressure_agreement`` divides each pattern, the simulated
taxel forces and the glove's pressure above its rest baseline (clip key ``taxel_load``), by its own total plus a
small floor, and sums the overlap ``min(p, q)`` over the taxels plus over the floor's share (the "nothing" bin).
It is 1 when the hand presses where the glove does in the same proportions (at any overall strength), also when
both press on nothing; it falls when pressure sits on other taxels, is distributed differently, or when one side
presses and the other does not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import mujoco
import numpy as np
import torch
import torch.nn.functional as F

from supermimic.kinematics.rotations import quat_apply, quat_apply_inverse
from supermimic.robots import HANDS_XML, ROBOT

PROPS_DIR = Path(__file__).resolve().parents[3] / "data/props"
PROP_SOURCES = {"broom": "cleaningtools_broom", "dustpan": "cleaningtools_dustpan"}
NEAR = 0.005  # m: taxels nearer the prop than this take most of their body's force
SOFT = 0.002  # m: width of the smooth proximity weight
NTAXEL = 129
OFF_TAXEL_SHARE = 0.02  # uniform weight of every taxel of a body when spreading its force
SIM_PRESSURE_FLOOR = 0.5  # N: "nothing" share of the simulated pattern
GLOVE_PRESSURE_FLOOR = 10.0  # raw glove units: same for the glove (a holding glove reads a few hundred in total)


def body_names() -> list[list[str]]:
    """Per hand of ``ROBOT.grip``, the bodies that carry taxels (sorted by body id), padded to the same length by
    repeating the last one (its slot then receives no taxels)."""
    from supermimic.data.glove_taxels import glove_layout

    m = mujoco.MjModel.from_xml_path(str(HANDS_XML))
    out = []
    for side, _ in ROBOT.grip:
        out.append([m.body(b).name for b in sorted(set(int(b) for b in glove_layout(m, side).body))])
    nb = max(len(n) for n in out)
    return [n + [n[-1]] * (nb - len(n)) for n in out]


class GloveModel:
    """Taxel placement and held-prop distance fields for the hands of ``ROBOT.grip`` (in that order)."""

    def __init__(self, body_indices: Callable[[list[str]], torch.Tensor], device, dtype=torch.float32):
        from supermimic.data.glove_taxels import glove_layout

        m = mujoco.MjModel.from_xml_path(str(HANDS_XML))
        body, local, slot, self.sdf, self.lo, self.span, self.prop_index = [], [], [], [], [], [], []
        self.force_body_names = body_names()
        for h, (side, prop) in enumerate(ROBOT.grip):
            lay = glove_layout(m, side)
            names = [m.body(int(b)).name for b in lay.body]
            body.append(body_indices(names))
            slot.append(torch.tensor([self.force_body_names[h].index(n) for n in names], device=device))
            local.append(torch.as_tensor(lay.pos, dtype=dtype, device=device))
            z = np.load(PROPS_DIR / PROP_SOURCES[prop] / "sdf.npz")
            g = torch.as_tensor(z["sdf"].astype(np.float32), device=device)  # (nx, ny, nz)
            self.sdf.append(g.permute(2, 1, 0)[None, None].to(dtype))  # (1, 1, nz, ny, nx) for grid_sample
            self.lo.append(torch.as_tensor(z["lo"], dtype=dtype, device=device))
            self.span.append(torch.tensor([float(z["voxel"]) * (n - 1) for n in g.shape], dtype=dtype, device=device))
            self.prop_index.append(ROBOT.props.index(prop))
        self.body = torch.stack(body)  # (H, 129)
        self.local = torch.stack(local)  # (H, 129, 3)
        self.slot = torch.stack(slot)  # (H, 129) index of each taxel's body among the force bodies
        self.num_bodies = len(self.force_body_names[0])
        # (H, 129, B) one-hot taxel -> force body
        self.onehot = F.one_hot(self.slot, self.num_bodies).to(dtype)

    def taxels_world(self, body_pos: torch.Tensor, body_quat: torch.Tensor) -> torch.Tensor:
        """All-body poses (…, NB, 3/4) -> taxel positions (…, H, 129, 3)."""
        p = body_pos[..., self.body, :]
        q = body_quat[..., self.body, :]
        return p + quat_apply(q, self.local.expand(*p.shape))

    def distances(self, body_pos: torch.Tensor, body_quat: torch.Tensor, prop_pos: torch.Tensor, prop_quat: torch.Tensor) -> torch.Tensor:
        """Signed distance (…, H, 129) of each hand's taxels to its held prop; props (…, P, 3/4)."""
        x = self.taxels_world(body_pos, body_quat)
        out = []
        for h, k in enumerate(self.prop_index):
            pq = prop_quat[..., k, None, :].expand(*x.shape[:-3], NTAXEL, 4)
            local = quat_apply_inverse(pq, x[..., h, :, :] - prop_pos[..., k, None, :])
            u = 2.0 * (local - self.lo[h]) / self.span[h] - 1.0  # (…, 129, 3) as (x, y, z) in [-1, 1]
            d = F.grid_sample(self.sdf[h], u.reshape(1, 1, 1, -1, 3), mode="bilinear", padding_mode="border", align_corners=True)
            out.append(d.reshape(*x.shape[:-3], NTAXEL))
        return torch.stack(out, dim=-2)

    @staticmethod
    def proximity(d: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((NEAR - d) / SOFT)

    def taxel_forces(self, d: torch.Tensor, body_force: torch.Tensor) -> torch.Tensor:
        """Per-taxel force (…, H, 129) from the force bodies' normal forces (…, H, B), spread by proximity (``d``: …, H, 129)."""
        w = self.proximity(d) + OFF_TAXEL_SHARE
        per_body = torch.einsum("...ht,htb->...hb", w, self.onehot)
        share = w / torch.einsum("...hb,htb->...ht", per_body, self.onehot)
        return share * torch.einsum("...hb,htb->...ht", body_force, self.onehot)

    @staticmethod
    def pressure_agreement(force: torch.Tensor, load: torch.Tensor) -> torch.Tensor:
        """Relative overlap (…, H) in [0, 1] of the simulated taxel forces (…, H, 129, N) with the glove's pressure
        (…, H, 129, raw units)."""
        p = force / (force.sum(-1, keepdim=True) + SIM_PRESSURE_FLOOR)
        q = load / (load.sum(-1, keepdim=True) + GLOVE_PRESSURE_FLOOR)
        return torch.minimum(p, q).sum(-1) + torch.minimum(1.0 - p.sum(-1), 1.0 - q.sum(-1))

"""Glove-pattern grip term of the capsule hands (torch, differentiable).

Each hand's 129 glove taxels sit at fixed points of its bodies (``supermimic.data.glove_taxels``, the layout fitted
to the glove's pressed pattern). A taxel *touches* the prop it holds (``ROBOT.grip``) with a smooth indicator of its
signed distance to the prop surface, ``sigmoid((TOUCH - d) / SOFT)``, the same contact model the layout fit used;
``d`` is read from the prop's signed-distance grid (scripts/assets/build_prop_sdf.py) by trilinear interpolation.

The grip is scored on force. The simulator measures the contact normal force between each taxel-carrying body
(``body_names``, the *force bodies*) and the held prop; ``taxel_forces`` spreads a body's force over its taxels in
proportion to how much each touches (plus a small uniform share, so a contact off the taxels still counts). The
reference is the glove's load above its rest baseline (clip key ``taxel_load``) times ``NEWTONS_PER_UNIT``.
``force_agreement`` is the overlap ``(2 sum min(F, R) + C) / (sum F + sum R + C)``: 1 when every taxel carries the
glove's force (also when the hand is free and pushes on nothing), lower when the hand squeezes too hard, too
softly or in other places than the glove.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import mujoco
import numpy as np
import torch
import torch.nn.functional as F

from supermimic.kinematics.rotations import quat_apply, quat_apply_inverse
from supermimic.robots import ASSETS, ROBOT

HANDS_XML = ASSETS / "capsule_hands/capsule_hands.xml"
PROPS_DIR = Path(__file__).resolve().parents[3] / "data/props"
PROP_SOURCES = {"broom": "cleaningtools_broom", "dustpan": "cleaningtools_dustpan"}
TOUCH = 0.005  # m: closer than this counts as touching
SOFT = 0.002  # m: width of the smooth indicator
NTAXEL = 129
# The glove reads raw units, not newtons. Calibrated once on this simulator: the median holding load of each glove
# (392 and 315 units) against the median contact normal force with which the trained policy (logs/capsule_grip_s3,
# iteration 7000) carries the broom and dustpan (16.7 and 11.3 N); scripts/diag_grip_forces.py.
NEWTONS_PER_UNIT = 0.04
OFF_TAXEL_SHARE = 0.02  # uniform weight of every taxel of a body when spreading its force
FORCE_SMOOTH = 1.0  # N, C of force_agreement


def body_names() -> list[list[str]]:
    """Per hand of ``ROBOT.grip``, the bodies that carry taxels (sorted by body id), padded to the same length by
    repeating the last one (its slot then receives no taxels)."""
    from supermimic.data.glove_taxels import fitted_adjust, taxel_layout

    m = mujoco.MjModel.from_xml_path(str(HANDS_XML))
    out = []
    for side, _ in ROBOT.grip:
        out.append([m.body(b).name for b in sorted(set(int(b) for b in taxel_layout(m, side, fitted_adjust()).body))])
    nb = max(len(n) for n in out)
    return [n + [n[-1]] * (nb - len(n)) for n in out]


class GloveModel:
    """Taxel placement and held-prop distance fields for the hands of ``ROBOT.grip`` (in that order)."""

    def __init__(self, body_indices: Callable[[list[str]], torch.Tensor], device, dtype=torch.float32):
        from supermimic.data.glove_taxels import fitted_adjust, taxel_layout

        m = mujoco.MjModel.from_xml_path(str(HANDS_XML))
        adjust = fitted_adjust()
        body, local, slot, self.sdf, self.lo, self.span, self.prop_index = [], [], [], [], [], [], []
        self.force_body_names = body_names()
        for h, (side, prop) in enumerate(ROBOT.grip):
            lay = taxel_layout(m, side, adjust)
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
    def touching(d: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((TOUCH - d) / SOFT)

    def taxel_forces(self, d: torch.Tensor, body_force: torch.Tensor) -> torch.Tensor:
        """Per-taxel force (…, H, 129) from the force bodies' normal forces (…, H, B), spread by touch (``d``: …, H, 129)."""
        w = self.touching(d) + OFF_TAXEL_SHARE
        per_body = torch.einsum("...ht,htb->...hb", w, self.onehot)
        share = w / torch.einsum("...hb,htb->...ht", per_body, self.onehot)
        return share * torch.einsum("...hb,htb->...ht", body_force, self.onehot)

    @staticmethod
    def force_agreement(force: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Overlap (…, H) of the taxel forces (…, H, 129) with the glove's (…, H, 129), both in newtons."""
        return (2.0 * torch.minimum(force, ref).sum(-1) + FORCE_SMOOTH) / (force.sum(-1) + ref.sum(-1) + FORCE_SMOOTH)

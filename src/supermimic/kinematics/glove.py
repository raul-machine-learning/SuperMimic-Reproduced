"""Glove-pattern grip term of the capsule hands (torch, differentiable).

Each hand's 129 glove taxels sit at fixed points of its bodies (``supermimic.data.glove_taxels``, the layout fitted
to the glove's pressed pattern). A taxel *touches* the prop it holds (``ROBOT.grip``) with a smooth indicator of its
signed distance to the prop surface, ``sigmoid((TOUCH - d) / SOFT)``, the same contact model the layout fit used;
``d`` is read from the prop's signed-distance grid (scripts/assets/build_prop_sdf.py) by trilinear interpolation.

The agreement with the glove is a smoothed Dice score between the touching taxels ``c`` and the taxels the glove reads
as pressed in the reference ``y``: ``(2 sum(y c) + 1) / (sum(y) + sum(c) + 1)``, 1 when they coincide (also when the
hand is free and touches nothing), toward 0 when the hand touches elsewhere or not where the glove is pressed.
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


class GloveModel:
    """Taxel placement and held-prop distance fields for the hands of ``ROBOT.grip`` (in that order)."""

    def __init__(self, body_indices: Callable[[list[str]], torch.Tensor], device, dtype=torch.float32):
        from supermimic.data.glove_taxels import fitted_adjust, taxel_layout

        m = mujoco.MjModel.from_xml_path(str(HANDS_XML))
        adjust = fitted_adjust()
        body, local, self.sdf, self.lo, self.span, self.prop_index = [], [], [], [], [], []
        for side, prop in ROBOT.grip:
            lay = taxel_layout(m, side, adjust)
            body.append(body_indices([m.body(int(b)).name for b in lay.body]))
            local.append(torch.as_tensor(lay.pos, dtype=dtype, device=device))
            z = np.load(PROPS_DIR / PROP_SOURCES[prop] / "sdf.npz")
            g = torch.as_tensor(z["sdf"].astype(np.float32), device=device)  # (nx, ny, nz)
            self.sdf.append(g.permute(2, 1, 0)[None, None].to(dtype))  # (1, 1, nz, ny, nx) for grid_sample
            self.lo.append(torch.as_tensor(z["lo"], dtype=dtype, device=device))
            self.span.append(torch.tensor([float(z["voxel"]) * (n - 1) for n in g.shape], dtype=dtype, device=device))
            self.prop_index.append(ROBOT.props.index(prop))
        self.body = torch.stack(body)  # (H, 129)
        self.local = torch.stack(local)  # (H, 129, 3)

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

    @staticmethod
    def agreement(touch: torch.Tensor, pressed: torch.Tensor) -> torch.Tensor:
        """Smoothed Dice (…, H) of touching (…, H, 129) against the glove's pressed taxels (…, H, 129)."""
        return (2.0 * (touch * pressed).sum(-1) + 1.0) / (touch.sum(-1) + pressed.sum(-1) + 1.0)

"""Wrist pose errors of the capsule-hand rig, straight from the wrist joints (``ROBOT.wrist_joints``).

Each wrist hangs from the welded root through slides along world x, y, z and hinges about its own y, x, z axes
(intrinsic, zero at a fixed base pose), so the position error is the slide difference and the orientation error
the angle of ``R_ref^-1 R`` with ``R = Ry Rx Rz``: the base pose cancels and no FK is needed.
"""

from __future__ import annotations

from functools import cache

import torch

from supermimic.kinematics.rotations import quat_from_angle_axis, quat_inv, quat_mul
from supermimic.robots import ROBOT


@cache
def wrist_joint_index(device) -> torch.Tensor:
    """(H, 6) joint indices of each hand's wrist joints."""
    return torch.tensor([ROBOT.joint_indices(w) for w in ROBOT.wrist_joints], dtype=torch.long, device=device)


def _wrist_quat(a: torch.Tensor) -> torch.Tensor:
    """a (…, 3) hinge angles about y, x, z (intrinsic) -> quat wxyz (…, 4)."""
    eye = torch.eye(3, dtype=a.dtype, device=a.device)
    q = [quat_from_angle_axis(a[..., k], eye[i].expand(*a.shape[:-1], 3)) for k, i in enumerate((1, 0, 2))]
    return quat_mul(quat_mul(q[0], q[1]), q[2])


def wrist_errors(q: torch.Tensor, q_ref: torch.Tensor, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint positions (…, NJ) -> wrist position error (…, H) in m and orientation error (…, H) in rad."""
    w, w_ref = q[..., index], q_ref[..., index]  # (…, H, 6)
    d = w[..., :3] - w_ref[..., :3]
    pos = torch.sqrt((d * d).sum(-1) + 1e-12)
    rel = quat_mul(quat_inv(_wrist_quat(w_ref[..., 3:])), _wrist_quat(w[..., 3:]))
    v = torch.sqrt((rel[..., 1:] ** 2).sum(-1) + 1e-12)
    ang = 2.0 * torch.atan2(v, rel[..., 0].abs())
    return pos, ang

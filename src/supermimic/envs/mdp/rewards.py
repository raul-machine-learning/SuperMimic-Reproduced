"""Analytic BeyondMimic tracking reward in torch (plan D4/D7).

One differentiable implementation shared by the simulator side (tests / logging)
and the imagination side (world-model reward head). Mirrors mjlab's
``tasks/tracking/mdp/rewards.py`` term by term, including the re-anchoring of the
reference bodies to the robot anchor in xy + yaw only
(``MotionCommand.update_relative_body_poses``).

Inputs are world-frame link quantities of the *tracked* bodies (same ordering as
``TRACKED_BODIES``), the anchor being one of them (``anchor_index``).
Rewards are returned already weighted and multiplied by ``dt`` like mjlab's reward
manager (``scale_rewards_by_dt=True``).

Robots with hands (``ROBOT.hand_joint_names``) get one more term, ``motion_hand_joint_pos``:
``exp(-mean_j (q_j - q_ref_j)^2 / std^2)`` over the actuated hand joints (the mimic followers are
determined by them), implemented for the simulator by :func:`motion_hand_joint_position_error_exp`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_box_minus, quat_inv, quat_mul, yaw_quat

from supermimic.robots import ROBOT

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _robot_default(name: str, value: float):
    """Field default, unless the robot overrides it (``ROBOT.reward_overrides``)."""
    return field(default_factory=lambda: dict(ROBOT.reward_overrides).get(name, value))


@dataclass
class TrackingRewardCfg:
    std_body_pos: float = _robot_default("std_body_pos", 0.3)
    std_body_ori: float = _robot_default("std_body_ori", 0.4)
    std_body_lin_vel: float = _robot_default("std_body_lin_vel", 1.0)
    std_body_ang_vel: float = _robot_default("std_body_ang_vel", 3.14)
    std_root_pos: float = 0.3
    std_root_ori: float = 0.4
    w_body_pos: float = 1.0
    w_body_ori: float = 1.0
    w_body_lin_vel: float = 1.0
    w_body_ang_vel: float = 1.0
    w_root_pos: float = 0.5
    w_root_ori: float = 0.5
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    std_hand_joint_pos: float = 0.3
    w_hand_joint_pos: float = field(default_factory=lambda: 1.0 if ROBOT.hand_joint_names else 0.0)
    dt: float = 0.02

    @property
    def max_step_reward(self) -> float:
        return self.dt * (self.w_body_pos + self.w_body_ori + self.w_body_lin_vel + self.w_body_ang_vel + self.w_root_pos + self.w_root_ori + self.w_hand_joint_pos)


def reanchor_reference(
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
    ref_anchor_pos_w: torch.Tensor,
    ref_anchor_quat_w: torch.Tensor,
    robot_anchor_pos_w: torch.Tensor,
    robot_anchor_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express the reference bodies in a frame that shares the robot anchor's xy position and yaw.

    Exactly mjlab's ``update_relative_body_poses``: z, roll and pitch stay those of the reference.
    Shapes: bodies (N, B, ·), anchors (N, ·).
    """
    nb = ref_body_pos_w.shape[-2]
    delta_pos = robot_anchor_pos_w.clone()
    delta_pos = torch.cat([delta_pos[..., :2], ref_anchor_pos_w[..., 2:3]], dim=-1)  # (N,3)
    delta_ori = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(ref_anchor_quat_w)))  # (N,4)
    delta_ori_b = delta_ori[..., None, :].expand(*delta_ori.shape[:-1], nb, 4)
    rel_quat = quat_mul(delta_ori_b, ref_body_quat_w)
    rel_pos = delta_pos[..., None, :] + quat_apply(delta_ori_b, ref_body_pos_w - ref_anchor_pos_w[..., None, :])
    return rel_pos, rel_quat


def _sq_quat_err(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Squared geodesic angle between quaternions, smooth at zero (== quat_error_magnitude**2)."""
    v = quat_box_minus(q1, q2)
    return (v * v).sum(-1)


def tracking_reward_terms(
    *,
    robot_body_pos_w: torch.Tensor,
    robot_body_quat_w: torch.Tensor,
    robot_body_lin_vel_w: torch.Tensor,
    robot_body_ang_vel_w: torch.Tensor,
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
    ref_body_lin_vel_w: torch.Tensor,
    ref_body_ang_vel_w: torch.Tensor,
    anchor_index: int,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    joint_pos: torch.Tensor,  # (N, L) joints under the limit penalty
    soft_limits: torch.Tensor,  # (L, 2) or (N, L, 2)
    cfg: TrackingRewardCfg,
    hand_joint_pos: torch.Tensor | None = None,  # (N, H)
    ref_hand_joint_pos: torch.Tensor | None = None,  # (N, H)
) -> dict[str, torch.Tensor]:
    """Per-term weighted, dt-scaled rewards (N,) plus 'total'."""
    ra_pos = robot_body_pos_w[..., anchor_index, :]
    ra_quat = robot_body_quat_w[..., anchor_index, :]
    ma_pos = ref_body_pos_w[..., anchor_index, :]
    ma_quat = ref_body_quat_w[..., anchor_index, :]
    rel_pos, rel_quat = reanchor_reference(ref_body_pos_w, ref_body_quat_w, ma_pos, ma_quat, ra_pos, ra_quat)

    e_root_pos = ((ma_pos - ra_pos) ** 2).sum(-1)
    e_root_ori = _sq_quat_err(ma_quat, ra_quat)
    e_body_pos = ((rel_pos - robot_body_pos_w) ** 2).sum(-1).mean(-1)
    e_body_ori = _sq_quat_err(rel_quat, robot_body_quat_w).mean(-1)
    e_body_lin = ((ref_body_lin_vel_w - robot_body_lin_vel_w) ** 2).sum(-1).mean(-1)
    e_body_ang = ((ref_body_ang_vel_w - robot_body_ang_vel_w) ** 2).sum(-1).mean(-1)

    lo = soft_limits[..., 0]
    hi = soft_limits[..., 1]
    out = (-(joint_pos - lo)).clamp(min=0.0) + (joint_pos - hi).clamp(min=0.0)
    joint_limit = out.sum(-1)
    action_rate = ((action - prev_action) ** 2).sum(-1)

    dt = cfg.dt
    terms = {
        "motion_global_root_pos": dt * cfg.w_root_pos * torch.exp(-e_root_pos / cfg.std_root_pos**2),
        "motion_global_root_ori": dt * cfg.w_root_ori * torch.exp(-e_root_ori / cfg.std_root_ori**2),
        "motion_body_pos": dt * cfg.w_body_pos * torch.exp(-e_body_pos / cfg.std_body_pos**2),
        "motion_body_ori": dt * cfg.w_body_ori * torch.exp(-e_body_ori / cfg.std_body_ori**2),
        "motion_body_lin_vel": dt * cfg.w_body_lin_vel * torch.exp(-e_body_lin / cfg.std_body_lin_vel**2),
        "motion_body_ang_vel": dt * cfg.w_body_ang_vel * torch.exp(-e_body_ang / cfg.std_body_ang_vel**2),
        "action_rate_l2": dt * cfg.w_action_rate * action_rate,
        "joint_limit": dt * cfg.w_joint_limit * joint_limit,
    }
    if hand_joint_pos is not None:
        e_hand = ((hand_joint_pos - ref_hand_joint_pos) ** 2).mean(-1)
        terms["motion_hand_joint_pos"] = dt * cfg.w_hand_joint_pos * torch.exp(-e_hand / cfg.std_hand_joint_pos**2)
    total = terms["motion_global_root_pos"]
    for k, v in terms.items():
        if k != "motion_global_root_pos":
            total = total + v
    terms["total"] = total
    return terms


def motion_hand_joint_position_error_exp(env: ManagerBasedRlEnv, command_name: str, std: float, joint_ids: list[int]) -> torch.Tensor:
    """Simulator side of ``motion_hand_joint_pos`` (unweighted, not dt-scaled, like mjlab's reward functions)."""
    from supermimic.envs.mdp.commands import MultiClipMotionCommand

    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    err = ((cmd.robot_joint_pos[:, joint_ids] - cmd.joint_pos[:, joint_ids]) ** 2).mean(-1)
    return torch.exp(-err / std**2)

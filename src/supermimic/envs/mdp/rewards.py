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
Robots with props add ``motion_prop_pos`` / ``motion_prop_ori`` (world pose of each prop against its reference),
and robots with a grip (``ROBOT.grip``) ``grip_force``, the agreement of the hands' contact forces on the props,
per taxel, with the glove's (``supermimic.kinematics.glove``). The simulator measures those forces with contact
sensors; the world model predicts them with its force head.
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
    std_prop_pos: float = 0.05
    std_prop_ori: float = 0.3
    w_prop_pos: float = field(default_factory=lambda: 1.0 if ROBOT.props else 0.0)
    w_prop_ori: float = field(default_factory=lambda: 0.5 if ROBOT.props else 0.0)
    w_grip: float = field(default_factory=lambda: 1.0 if ROBOT.grip else 0.0)
    dt: float = 0.02

    @property
    def max_step_reward(self) -> float:
        return self.dt * (self.w_body_pos + self.w_body_ori + self.w_body_lin_vel + self.w_body_ang_vel + self.w_root_pos + self.w_root_ori
                          + self.w_hand_joint_pos + self.w_prop_pos + self.w_prop_ori + self.w_grip)


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


def weighted_mean(e: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Mean over the last axis with weights; 0 when every weight is 0 (nothing left to track)."""
    return (e * w).sum(-1) / w.sum(-1).clamp(min=1e-6)


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
    prop_pos_w: torch.Tensor | None = None,  # (N, P, 3)
    prop_quat_w: torch.Tensor | None = None,  # (N, P, 4)
    ref_prop_pos_w: torch.Tensor | None = None,
    ref_prop_quat_w: torch.Tensor | None = None,
    grip_agreement: torch.Tensor | None = None,  # (N, G) per grip hand, in [0, 1]
    body_weight: torch.Tensor | None = None,  # (N, B) 0 for bodies released from tracking (grip_assist.TrackingRelease)
    hand_weight: torch.Tensor | None = None,  # (N, H) same for the hand joints
) -> dict[str, torch.Tensor]:
    """Per-term weighted, dt-scaled rewards (N,) plus 'total'."""
    body_mean = (lambda e: e.mean(-1)) if body_weight is None else (lambda e: weighted_mean(e, body_weight))
    ra_pos = robot_body_pos_w[..., anchor_index, :]
    ra_quat = robot_body_quat_w[..., anchor_index, :]
    ma_pos = ref_body_pos_w[..., anchor_index, :]
    ma_quat = ref_body_quat_w[..., anchor_index, :]
    rel_pos, rel_quat = reanchor_reference(ref_body_pos_w, ref_body_quat_w, ma_pos, ma_quat, ra_pos, ra_quat)

    e_root_pos = ((ma_pos - ra_pos) ** 2).sum(-1)
    e_root_ori = _sq_quat_err(ma_quat, ra_quat)
    e_body_pos = body_mean(((rel_pos - robot_body_pos_w) ** 2).sum(-1))
    e_body_ori = body_mean(_sq_quat_err(rel_quat, robot_body_quat_w))
    e_body_lin = body_mean(((ref_body_lin_vel_w - robot_body_lin_vel_w) ** 2).sum(-1))
    e_body_ang = body_mean(((ref_body_ang_vel_w - robot_body_ang_vel_w) ** 2).sum(-1))

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
        e = (hand_joint_pos - ref_hand_joint_pos) ** 2
        e_hand = e.mean(-1) if hand_weight is None else weighted_mean(e, hand_weight)
        terms["motion_hand_joint_pos"] = dt * cfg.w_hand_joint_pos * torch.exp(-e_hand / cfg.std_hand_joint_pos**2)
    if prop_pos_w is not None:
        e_prop_pos = ((prop_pos_w - ref_prop_pos_w) ** 2).sum(-1).mean(-1)
        e_prop_ori = _sq_quat_err(ref_prop_quat_w, prop_quat_w).mean(-1)
        terms["motion_prop_pos"] = dt * cfg.w_prop_pos * torch.exp(-e_prop_pos / cfg.std_prop_pos**2)
        terms["motion_prop_ori"] = dt * cfg.w_prop_ori * torch.exp(-e_prop_ori / cfg.std_prop_ori**2)
    if grip_agreement is not None:
        terms["grip_force"] = dt * cfg.w_grip * grip_agreement.mean(-1)
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
    e = (cmd.robot_joint_pos[:, joint_ids] - cmd.joint_pos[:, joint_ids]) ** 2
    err = e.mean(-1) if not ROBOT.grip else weighted_mean(e, tracking_release_weights(cmd)[1])
    return torch.exp(-err / std**2)


def tracking_release_weights(cmd) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, B) tracked-body and (N, H) hand-joint weights of the current frame (``grip_assist.TrackingRelease``)."""
    rel = getattr(cmd, "_release", None)
    if rel is None:
        from supermimic.envs.mdp.grip_assist import TrackingRelease

        rel = cmd._release = TrackingRelease(ROBOT.tracked_bodies, ROBOT.hand_joint_names, cmd.device)
    return rel.weights(cmd.library.grip_hold_early[cmd.frame])


def _released_body_term(env: ManagerBasedRlEnv, command_name: str, std: float, error) -> torch.Tensor:
    from supermimic.envs.mdp.commands import MultiClipMotionCommand

    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return torch.exp(-weighted_mean(error(cmd), tracking_release_weights(cmd)[0]) / std**2)


# mjlab's body tracking terms with the finger bodies of holding hands left out (robots with a grip).
def motion_body_position_released_exp(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
    return _released_body_term(env, command_name, std, lambda c: ((c.body_pos_relative_w - c.robot_body_pos_w) ** 2).sum(-1))


def motion_body_orientation_released_exp(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
    return _released_body_term(env, command_name, std, lambda c: _sq_quat_err(c.body_quat_relative_w, c.robot_body_quat_w))


def motion_body_lin_vel_released_exp(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
    return _released_body_term(env, command_name, std, lambda c: ((c.body_lin_vel_w - c.robot_body_lin_vel_w) ** 2).sum(-1))


def motion_body_ang_vel_released_exp(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
    return _released_body_term(env, command_name, std, lambda c: ((c.body_ang_vel_w - c.robot_body_ang_vel_w) ** 2).sum(-1))


def motion_prop_position_error_exp(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
    """Simulator side of ``motion_prop_pos``: ``exp(-mean_k |p_k - p_ref_k|^2 / std^2)`` over the props."""
    from supermimic.envs.mdp.commands import MultiClipMotionCommand

    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return torch.exp(-((cmd.sim_prop_pos_w - cmd.prop_pos_w) ** 2).sum(-1).mean(-1) / std**2)


def motion_prop_orientation_error_exp(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
    from supermimic.envs.mdp.commands import MultiClipMotionCommand

    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return torch.exp(-_sq_quat_err(cmd.prop_quat_w, cmd.sim_prop_quat_w).mean(-1) / std**2)


def sim_glove(cmd):
    """``GloveModel`` of the simulated robot, built once per command term."""
    from supermimic.kinematics.glove import GloveModel

    glove = getattr(cmd, "_glove", None)
    if glove is None:
        names = cmd.robot.body_names
        glove = cmd._glove = GloveModel(lambda b: torch.tensor([names.index(n) for n in b], device=cmd.device), cmd.device)
    return glove


def sim_taxel_forces(env: ManagerBasedRlEnv, cmd) -> torch.Tensor:
    """(N, G, 129) taxel forces (N) of the simulated hands: the contact sensors' body forces spread by touch."""
    from supermimic.envs.mdp.grip_force import body_forces

    glove = sim_glove(cmd)
    d = glove.distances(cmd.robot.data.body_link_pos_w, cmd.robot.data.body_link_quat_w, cmd.sim_prop_pos_w, cmd.sim_prop_quat_w)
    return glove.taxel_forces(d, body_forces(env))


def grip_force_agreement_sim(env: ManagerBasedRlEnv, cmd) -> torch.Tensor:
    """(N, G) force agreement of the simulated hands with the glove (``GloveModel.force_agreement``)."""
    return sim_glove(cmd).force_agreement(sim_taxel_forces(env, cmd), cmd.taxel_force)


def grip_force_agreement(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Simulator side of ``grip_force``: the mean over the grip hands of the force agreement."""
    from supermimic.envs.mdp.commands import MultiClipMotionCommand

    return grip_force_agreement_sim(env, cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))).mean(-1)

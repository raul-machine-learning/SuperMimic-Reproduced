"""Analytic heads of the world model (plan D4): observation, reward, termination.

Everything here is a pure differentiable function of the 71-d simulator state
``s = [qpos(36) | qvel(35)]`` (root position expressed relative to the env origin,
see ``StateIO``), the previous clean action, the sim context ``xi`` and the
reference frame, so the same code serves the simulator-side tests and the
imagined rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.envs.mdp.rewards import TrackingRewardCfg, tracking_reward_terms
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import projected_gravity, quat_apply_inverse
from supermimic.robots import ROBOT

OBS_DIM = ROBOT.obs_frame_dim  # 93 for the G1
# Term layout of the ``policy`` observation group (order matters: term-major history).
OBS_TERMS = (("joint_pos", ROBOT.nj), ("joint_vel", ROBOT.nj), ("base_ang_vel", 3), ("projected_gravity", 3), ("actions", ROBOT.na)) + (
    (("prop_pose", 9 * ROBOT.num_props),) if ROBOT.props else ()
)


def split_state(s: torch.Tensor):
    """-> root_pos (…,3), root_quat (…,4), joint_pos (…,NJ), root_lin_vel_w (…,3), root_ang_vel_b (…,3), joint_vel (…,NJ)."""
    return ROBOT.split_state(s)


def analytic_observation(
    s: torch.Tensor, prev_action: torch.Tensor, xi: torch.Tensor, default_joint_pos: torch.Tensor
) -> torch.Tensor:
    """Noise-free proprioceptive observation (paper Table A.1, 93-d for the G1) from state + context.

    xi[..., 1:1+NJ] is the encoder bias: observed joint positions are ``q + bias - q0``. With props, each prop's
    pose in the root frame follows (9 per prop).
    """
    root_pos, root_quat, q, _, ang_b, qd = split_state(s)
    bias = xi[..., 1 : 1 + ROBOT.nj]
    parts = [q + bias - default_joint_pos, qd, ang_b, projected_gravity(root_quat), prev_action]
    if ROBOT.props:
        from supermimic.envs.mdp.motion_library import prop_pose_in_root

        prop_pos, prop_quat = ROBOT.split_props(s)[:2]
        parts.append(prop_pose_in_root(root_pos, root_quat, prop_pos, prop_quat))
    return torch.cat(parts, dim=-1)


def anchor_relative_obs(robot_anchor_pos: torch.Tensor, robot_anchor_quat: torch.Tensor, ref_anchor_pos: torch.Tensor, ref_anchor_quat: torch.Tensor) -> torch.Tensor:
    """Reference anchor pose expressed in the robot anchor frame: [pos (3), rot6d (6)] (= mjlab's
    motion_anchor_pos_b / motion_anchor_ori_b, plan D5 alternative)."""
    from supermimic.kinematics.rotations import matrix_from_quat, quat_inv, quat_mul

    q10 = quat_inv(robot_anchor_quat)
    pos = quat_apply_inverse(robot_anchor_quat, ref_anchor_pos - robot_anchor_pos)
    q12 = quat_mul(q10, ref_anchor_quat)
    mat = matrix_from_quat(q12)
    return torch.cat([pos, mat[..., :2].reshape(*q12.shape[:-1], 6)], dim=-1)


ANCHOR_OBS_DIM = 9


def flatten_history_term_major(hist: torch.Tensor) -> torch.Tensor:
    """(H, N, OBS_DIM) oldest→newest  ->  (N, H*OBS_DIM) in mjlab's term-major layout [A_t0..A_tH-1, B_t0..]."""
    H, N, _ = hist.shape
    chunks = []
    start = 0
    for _, d in OBS_TERMS:
        chunks.append(hist[:, :, start : start + d].permute(1, 0, 2).reshape(N, H * d))
        start += d
    return torch.cat(chunks, dim=-1)


@dataclass
class ReferenceFrame:
    """Reference quantities of one target frame for a batch (N, ·)."""

    body_pos_w: torch.Tensor  # (N, B, 3)
    body_quat_w: torch.Tensor  # (N, B, 4)
    body_lin_vel_w: torch.Tensor  # (N, B, 3)
    body_ang_vel_w: torch.Tensor  # (N, B, 3)
    root_pos_w: torch.Tensor  # (N, 3)
    root_quat_w: torch.Tensor  # (N, 4)
    joint_pos: torch.Tensor | None = None  # (N, NJ); needed by the hand reward only
    prop_pos_w: torch.Tensor | None = None  # (N, P, 3); robots with props
    prop_quat_w: torch.Tensor | None = None  # (N, P, 4)
    taxel_force: torch.Tensor | None = None  # (N, G, 129) glove force per taxel (N); robots with a grip
    grip_hold_early: torch.Tensor | None = None  # (N, G) anticipated holding flags (grip_assist.TrackingRelease)


class AnalyticHeads:
    """Reward / termination features computed from the predicted state through differentiable FK."""

    def __init__(
        self,
        fk: G1Kinematics,
        tracked_body_names: tuple[str, ...],
        anchor_body_name: str,
        root_body_name: str,
        soft_limits: torch.Tensor,
        default_joint_pos: torch.Tensor,
        reward_cfg: TrackingRewardCfg | None = None,
    ):
        """``soft_limits``: (NJ, 2) for every hinge; only ``ROBOT.limit_joint_names`` are penalised."""
        self.fk = fk
        self.tracked_idx = fk.body_indices(tracked_body_names)
        self.anchor_index = tracked_body_names.index(anchor_body_name)
        self.root_fk_index = fk.body_index(root_body_name)
        dev = soft_limits.device
        self.limit_idx = torch.tensor(ROBOT.joint_indices(ROBOT.limit_joint_names), device=dev)
        self.soft_limits = soft_limits[self.limit_idx]
        self.hand_idx = torch.tensor(ROBOT.joint_indices(ROBOT.hand_joint_names), device=dev) if ROBOT.hand_joint_names else None
        self.default_joint_pos = default_joint_pos
        self.reward_cfg = reward_cfg or TrackingRewardCfg()
        self.glove = self.release = None
        if ROBOT.grip:
            from supermimic.envs.mdp.grip_assist import TrackingRelease
            from supermimic.kinematics.glove import GloveModel

            self.glove = GloveModel(lambda names: torch.as_tensor(fk.body_indices(tuple(names)), device=dev), dev)
            self.release = TrackingRelease(tuple(tracked_body_names), ROBOT.hand_joint_names, dev)

    def robot_bodies(self, s: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.fk(s[..., : ROBOT.nq], s[..., ROBOT.nq : ROBOT.robot_state_dim])
        idx = self.tracked_idx
        return {
            "pos": out["pos"][..., idx, :],
            "quat": out["quat"][..., idx, :],
            "lin_vel": out["lin_vel"][..., idx, :],
            "ang_vel": out["ang_vel"][..., idx, :],
            "root_pos": out["pos"][..., self.root_fk_index, :],
            "root_quat": out["quat"][..., self.root_fk_index, :],
            "all_pos": out["pos"],
            "all_quat": out["quat"],
        }

    def grip_distances(self, s: torch.Tensor, bodies) -> torch.Tensor:
        """(N, G, 129) signed distance of each gripping hand's taxels to its prop (``GloveModel.distances``)."""
        assert self.glove is not None
        prop_pos, prop_quat = ROBOT.split_props(s)[:2]
        return self.glove.distances(bodies["all_pos"], bodies["all_quat"], prop_pos, prop_quat)

    def grip_agreement(self, d: torch.Tensor, body_force: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        """(N, G) force agreement with the glove from the taxel distances ``d`` and the force bodies' normal forces
        (N, G, B): measured by the simulator or predicted by the world model's force head."""
        assert self.glove is not None
        return self.glove.force_agreement(self.glove.taxel_forces(d, body_force), ref.taxel_force)

    def reward(self, s_next: torch.Tensor, action: torch.Tensor, prev_action: torch.Tensor, ref: ReferenceFrame, bodies=None,
               grip: tuple[torch.Tensor, torch.Tensor] | None = None):
        """Reward r_{t+1} of reaching s_next (= s_{t+1}) under target frame g_t. Returns dict with 'total'.
        ``grip``: (taxel distances, force-body forces) of s_next, required for robots with a grip."""
        b = bodies if bodies is not None else self.robot_bodies(s_next)
        q = split_state(s_next)[2]
        hand = ref_hand = None
        if self.hand_idx is not None:
            hand, ref_hand = q[..., self.hand_idx], ref.joint_pos[..., self.hand_idx]
        props = {}
        if ROBOT.props:
            prop_pos, prop_quat = ROBOT.split_props(s_next)[:2]
            props = dict(prop_pos_w=prop_pos, prop_quat_w=prop_quat, ref_prop_pos_w=ref.prop_pos_w, ref_prop_quat_w=ref.prop_quat_w)
        if self.glove is not None:
            assert grip is not None, "robots with a grip need the contact forces of s_next"
            props["grip_agreement"] = self.grip_agreement(*grip, ref)
        if self.release is not None:
            props["body_weight"], props["hand_weight"] = self.release.weights(ref.grip_hold_early)
        return tracking_reward_terms(
            robot_body_pos_w=b["pos"],
            robot_body_quat_w=b["quat"],
            robot_body_lin_vel_w=b["lin_vel"],
            robot_body_ang_vel_w=b["ang_vel"],
            ref_body_pos_w=ref.body_pos_w,
            ref_body_quat_w=ref.body_quat_w,
            ref_body_lin_vel_w=ref.body_lin_vel_w,
            ref_body_ang_vel_w=ref.body_ang_vel_w,
            anchor_index=self.anchor_index,
            action=action,
            prev_action=prev_action,
            joint_pos=q[..., self.limit_idx],
            soft_limits=self.soft_limits,
            cfg=self.reward_cfg,
            hand_joint_pos=hand,
            ref_hand_joint_pos=ref_hand,
            **props,
        )

    @staticmethod
    def termination_features(s: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        """(N, 2), the quantities thresholded by ``ROBOT.termination_thresholds``.

        * ``root``: [height error, tilt error] of the free-joint root (the paper's 0.25 m, 0.8); needs ``ref.root_*``.
        * ``wrists``: [largest wrist position error (m), largest wrist orientation error (rad)]; needs ``ref.joint_pos``.
          With props, the first feature is the larger of that and the largest prop position error rescaled from
          ``ROBOT.prop_drop_threshold`` to the wrist threshold (needs ``ref.prop_pos_w``).
        """
        if ROBOT.termination == "wrists":
            from supermimic.kinematics.wrists import wrist_errors

            pos, ang = wrist_errors(split_state(s)[2], ref.joint_pos)
            pos = pos.amax(-1)
            if ROBOT.props:
                d = ROBOT.split_props(s)[0] - ref.prop_pos_w
                prop = torch.sqrt((d * d).sum(-1) + 1e-12).amax(-1)
                pos = torch.maximum(pos, prop * (ROBOT.termination_thresholds[0] / ROBOT.prop_drop_threshold))
            return torch.stack([pos, ang.amax(-1)], dim=-1)
        root_pos, root_quat = s[..., 0:3], s[..., 3:7]
        g = torch.tensor([0.0, 0.0, -1.0], dtype=root_quat.dtype, device=root_quat.device).expand(root_quat.shape[:-1] + (3,))
        tilt = quat_apply_inverse(ref.root_quat_w, g)[..., 2] - quat_apply_inverse(root_quat, g)[..., 2]
        height = ref.root_pos_w[..., 2] - root_pos[..., 2]
        return torch.stack([height, tilt], dim=-1)

    def terminated(self, s: torch.Tensor, ref: ReferenceFrame, thresholds: tuple[float, float] = ROBOT.termination_thresholds) -> torch.Tensor:
        f = self.termination_features(s, ref)
        return (f[..., 0].abs() > thresholds[0]) | (f[..., 1].abs() > thresholds[1])

    @staticmethod
    def root_error_features(s: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        """Critic-only privileged drift features (plan D10), (N, 4): root position error in the reference
        heading frame (3) and yaw error (1); for ``wrists`` termination, each wrist's position and orientation error."""
        from supermimic.kinematics.rotations import heading_yaw, quat_apply_inverse, wrap_to_pi, yaw_quat

        if ROBOT.termination == "wrists":
            from supermimic.kinematics.wrists import wrist_errors

            pos, ang = wrist_errors(split_state(s)[2], ref.joint_pos)
            return torch.stack([pos, ang], dim=-1).flatten(-2)
        dp = quat_apply_inverse(yaw_quat(ref.root_quat_w), s[..., 0:3] - ref.root_pos_w)
        dyaw = wrap_to_pi(heading_yaw(s[..., 3:7]) - heading_yaw(ref.root_quat_w))
        return torch.cat([dp, dyaw[..., None]], dim=-1)

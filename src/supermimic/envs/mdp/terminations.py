"""Paper terminations: root (pelvis) height deviation and tilt; clip end as time-out. Robots with
``ROBOT.termination == "wrists"`` (welded root) end on wrist position / orientation error instead."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from supermimic.envs.mdp.commands import MultiClipMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def bad_root_height(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.25) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return (cmd.root_pos_w[:, 2] - cmd.robot_root_pos_w[:, 2]).abs() > threshold


def bad_root_tilt(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.8) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    g = cmd.robot.data.gravity_vec_w
    ref = quat_apply_inverse(cmd.root_quat_w, g)[:, 2]
    rob = quat_apply_inverse(cmd.robot_root_quat_w, g)[:, 2]
    return (ref - rob).abs() > threshold


def bad_wrist_pos(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.1) -> torch.Tensor:
    """Some wrist farther than ``threshold`` (m) from its reference (capsule-hand rig)."""
    from supermimic.kinematics.wrists import wrist_errors, wrist_joint_index

    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    pos, _ = wrist_errors(cmd.robot_joint_pos, cmd.joint_pos, wrist_joint_index(cmd.device))
    return pos.amax(-1) > threshold


def bad_wrist_ori(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.8) -> torch.Tensor:
    """Some wrist turned more than ``threshold`` (rad) away from its reference (capsule-hand rig)."""
    from supermimic.kinematics.wrists import wrist_errors, wrist_joint_index

    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    _, ang = wrist_errors(cmd.robot_joint_pos, cmd.joint_pos, wrist_joint_index(cmd.device))
    return ang.amax(-1) > threshold


def clip_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return cmd.clip_end

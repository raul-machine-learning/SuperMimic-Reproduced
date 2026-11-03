from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from supermimic.envs.mdp.commands import MultiClipMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def goal_future_frames(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """(N, K*68) future reference-frame features (plan D5)."""
    command = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return command.goal


def prop_pose(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """(N, 9P) each simulated prop's [position (3), rot6d (6)] in the robot root frame."""
    from supermimic.envs.mdp.motion_library import prop_pose_in_root

    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return prop_pose_in_root(cmd.robot_root_pos_w, cmd.robot_root_quat_w, cmd.sim_prop_pos_w, cmd.sim_prop_quat_w)

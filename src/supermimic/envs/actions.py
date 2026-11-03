"""Joint-position action with execution noise (plan D9).

The *clean* policy action stays in ``raw_action`` (so ``last_action`` observations and
the action-rate penalty see it); Gaussian noise ``eps ~ N(0, std^2)`` is added to the
normalized action only when computing the executed set-point. ``executed_raw`` exposes
``a + eps`` for the replay buffer (the world model is fit on executed actions).

With ``reference_offset`` (robots with ``ROBOT.residual_actions``) the set-point is the reference joint position of
the next frame plus the scaled action, instead of the default pose plus the scaled action. ``executed_abs`` is
the executed action in default-offset units, ``a + eps + (q_ref - q0) / scale``: the replay buffer stores it, so
the world model sees the same action meaning either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class NoisyJointPositionAction(JointPositionAction):
    cfg: NoisyJointPositionActionCfg

    def __init__(self, cfg: NoisyJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._executed_raw = torch.zeros_like(self._raw_actions)
        self._executed_abs = torch.zeros_like(self._raw_actions)
        self.noise_std = float(cfg.noise_std)
        self._default_offset = self._offset.clone() if isinstance(self._offset, torch.Tensor) else self._offset

    @property
    def executed_raw(self) -> torch.Tensor:
        return self._executed_raw

    @property
    def executed_abs(self) -> torch.Tensor:
        return self._executed_abs

    def _reference_target(self) -> torch.Tensor:
        """(N, A) reference joint positions of the frame after the current one."""
        cmd = self._env.command_manager.get_term(self.cfg.command_name)
        lib = cmd.library
        nxt = torch.minimum(cmd.frame + 1, lib.last_frame_of(cmd.frame))
        return lib.ref(nxt).joint_pos[:, self._target_ids]

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        if self.noise_std > 0:
            self._executed_raw = self._raw_actions + self.noise_std * torch.randn_like(self._raw_actions)
        else:
            self._executed_raw = self._raw_actions.clone()
        if self.cfg.reference_offset:
            self._offset = self._reference_target()
        self._processed_actions = self._executed_raw * self._scale + self._offset
        self._executed_abs = self._executed_raw + (self._offset - self._default_offset) / self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._executed_raw[env_ids] = 0.0
        self._executed_abs[env_ids] = 0.0


@dataclass(kw_only=True)
class NoisyJointPositionActionCfg(JointPositionActionCfg):
    noise_std: float = 0.1
    reference_offset: bool = False
    command_name: str = "motion"

    def build(self, env: ManagerBasedRlEnv) -> NoisyJointPositionAction:
        return NoisyJointPositionAction(self, env)

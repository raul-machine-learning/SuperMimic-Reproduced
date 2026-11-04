"""One imagined step as a single compiled function (plan 3.6: launch-overhead control).

``ImagStep`` bundles world-model dynamics, analytic FK reward (with the force head's grip forces), termination logits and the
next noise-free observation into one call so ``torch.compile`` can fuse the hundreds of
small kernels of the per-body FK loop. Reference quantities are passed as plain tensors
(already gathered from the motion library) so the compiled graph has static shapes.
"""

from __future__ import annotations

import torch

from supermimic.robots import ROBOT
from supermimic.world_model.heads import ReferenceFrame, analytic_observation
from supermimic.world_model.world_model import WorldModel

REWARD_TERM_NAMES = (
    "motion_global_root_pos",
    "motion_global_root_ori",
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
    "action_rate_l2",
    "joint_limit",
) + (("motion_hand_joint_pos",) if ROBOT.hand_joint_names else ()) + (("motion_prop_pos", "motion_prop_ori") if ROBOT.props else ()) + (
    ("grip_force",) if ROBOT.grip else ()
)


class ImagStep:
    def __init__(self, wm: WorldModel, default_joint_pos: torch.Tensor, compile_mode: str | None = None):
        self.wm = wm
        self.q0 = default_joint_pos
        self._fn = self._step
        if compile_mode:
            mode = None if compile_mode == "default" else compile_mode
            self._fn = torch.compile(self._step, mode=mode, dynamic=False)

    def _step(self, s, a, a_prev, xi, a_offset, *ref_fields):
        """``a_offset``: with residual actions, the reference set-point in default-offset action units (else None)."""
        ref = ReferenceFrame(*ref_fields)
        a_set = a if a_offset is None else a + a_offset
        s_next = self.wm.step(s, a_set, xi)
        bodies = self.wm.heads.robot_bodies(s_next)
        grip = self.wm.grip_forces(s_next, a_set, bodies) if self.wm.force_head is not None else None
        terms = self.wm.heads.reward(s_next, a, a_prev, ref, bodies, grip=grip)
        r = terms["total"]
        term_vec = torch.stack([terms[k] for k in REWARD_TERM_NAMES], dim=0)  # (len(REWARD_TERM_NAMES), n)
        logits = self.wm.termination_logits(s_next, ref)
        obs_clean = analytic_observation(s_next, a, xi, self.q0)
        ai = self.wm.heads.anchor_index
        anchor_pose = torch.cat([bodies["pos"][..., ai, :], bodies["quat"][..., ai, :]], dim=-1)  # (n, 7)
        return s_next, r, term_vec, logits, obs_clean, anchor_pose

    def __call__(self, s, a, a_prev, xi, ref: ReferenceFrame, a_offset=None):
        return self._fn(s, a, a_prev, xi, a_offset, *vars(ref).values())

"""State representation for the world model (plan D1–D3).

Sizes below are for the G1 (NJ = 29 hinges); other robots (``supermimic.robots``) substitute their NJ.

Simulator state  s = [root_pos(3), root_quat wxyz(4), joint_pos(29), root_lin_vel_w(3),
                      root_ang_vel_b(3), joint_vel(29)]            (71, MuJoCo layout, origin-free)

Canonical features (network input, heading-invariant, 71):
    [root_z(1), rot6d(heading^-1 ⊗ root_quat)(6), root_lin_vel in heading frame(3),
     root_ang_vel_b(3), joint_pos(29), joint_vel(29)]

Network output ``delta`` (70) and the integration step (dt = 0.02 s):
    dv_h(3), dw_b(3), dqd(29)            velocity increments, heading / body frame
    dp_h(3), dth_b(3), dq(29)            position corrections on top of explicit integration
    v'  = v + R_h dv_h               w' = w + dw_b          qd' = qd + dqd
    p'  = p + dt v' + R_h dp_h       q' = q ⊗ exp(dt w' + dth_b)    qj' = qj + dt qd' + dq
With zero output the model extrapolates at constant velocity (a physical prior); the
last layer of the dynamics MLP is zero-initialised so training starts from it.

Loss / error normalisation (D3): per-quantity nominal scales sigma applied to the
70-d error vector [dp(3), rotvec(3), dv(3), dw(3), dq(29), dqd(29)].

Props (``ROBOT.props``) append per prop, after the robot block:
    features  [pos rel. root in heading frame(3), rot6d(heading^-1 ⊗ quat)(6), lin vel in heading frame(3), ang vel body(3)]
    delta     [dv_h(3), dw_b(3), dp_h(3), dth_b(3)], integrated like the root
    error     [dp(3), rotvec(3), dv(3), dw(3)]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from supermimic.kinematics.rotations import (
    heading_quat,
    quat_apply,
    quat_apply_inverse,
    quat_canonical,
    quat_from_rotvec,
    quat_mul,
    quat_normalize,
    rot6d_from_quat,
    rotvec_from_quat,
)
from supermimic.robots import ROBOT

STATE_DIM = ROBOT.state_dim
FEAT_DIM = ROBOT.feat_dim
NP = ROBOT.num_props
DELTA_DIM = ROBOT.delta_dim
NJ = ROBOT.nj


@dataclass
class StateSpecCfg:
    dt: float = 0.02
    # Nominal error scales (D3) for [root pos m, root rotvec rad, root lin vel m/s, root ang vel rad/s, joint pos rad, joint vel rad/s].
    err_scale: tuple[float, ...] = (0.05, 0.1, 0.5, 1.0, 0.1, 2.0)
    # Output scales of the dynamics head for [dv_h, dw_b, dqd, dp_h, dth_b, dq].
    out_scale: tuple[float, ...] = (0.3, 0.5, 2.0, 0.01, 0.02, 0.02)
    # Input feature scales for [root_z, rot6d, lin_vel_h, ang_vel_b, joint_pos, joint_vel] (divide by these).
    in_scale: tuple[float, ...] = (1.0, 1.0, 1.0, 2.0, 1.0, 5.0)
    # Per prop: error scales [pos, rotvec, lin vel, ang vel], output scales [dv_h, dw_b, dp_h, dth_b],
    # input scales [pos, rot6d, lin vel, ang vel].
    prop_err_scale: tuple[float, ...] = (0.02, 0.1, 0.5, 2.0)
    prop_out_scale: tuple[float, ...] = (0.3, 2.0, 0.005, 0.02)
    prop_in_scale: tuple[float, ...] = (0.5, 1.0, 1.0, 5.0)
    explicit_integration: bool = True
    # xi normalisation: (offset, scale) for friction, encoder bias, com offset.
    xi_offset: tuple[float, ...] = field(default_factory=lambda: (0.75,) + (0.0,) * NJ + (0.0,) * 3)
    xi_scale: tuple[float, ...] = field(default_factory=lambda: (0.45,) + (0.01,) * NJ + (0.05,) * 3)


class StateSpec:
    def __init__(self, cfg: StateSpecCfg, device, dtype=torch.float32):
        self.cfg = cfg
        self.dt = cfg.dt
        # Joint segments are scaled per joint (ROBOT.joint_scale: slides in metres are smaller than hinge angles).
        js = torch.tensor(ROBOT.joint_scale)
        rep = lambda scales, sizes: torch.cat([torch.full((n,), float(s)) * (js if n == NJ else 1.0) for s, n in zip(scales, sizes)])  # noqa: E731
        prop = lambda scales, sizes: torch.cat([torch.full((n,), float(s)) for s, n in zip(scales, sizes)]).repeat(NP)  # noqa: E731
        self.err_scale = torch.cat([rep(cfg.err_scale, (3, 3, 3, 3, NJ, NJ)), prop(cfg.prop_err_scale, (3, 3, 3, 3))]).to(device=device, dtype=dtype)
        self.out_scale = torch.cat([rep(cfg.out_scale, (3, 3, NJ, 3, 3, NJ)), prop(cfg.prop_out_scale, (3, 3, 3, 3))]).to(device=device, dtype=dtype)
        self.in_scale = torch.cat([rep(cfg.in_scale, (1, 6, 3, 3, NJ, NJ)), prop(cfg.prop_in_scale, (3, 6, 3, 3))]).to(device=device, dtype=dtype)
        self.xi_offset = torch.tensor(cfg.xi_offset, device=device, dtype=dtype)
        self.xi_scale = torch.tensor(cfg.xi_scale, device=device, dtype=dtype)

    # ------------------------------------------------------------------ split / features

    @staticmethod
    def split(s: torch.Tensor):
        return ROBOT.split_state(s)

    def canonical(self, s: torch.Tensor) -> torch.Tensor:
        """Heading-invariant, scaled network input features (…, 71)."""
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        hq = heading_quat(q)
        q_local = quat_canonical(quat_mul(quat_apply_inverse_quat(hq), q))
        v_h = quat_apply_inverse(hq, v)
        parts = [p[..., 2:3], rot6d_from_quat(q_local), v_h, w, qj, qd]
        if NP:
            pp, pq, pv, pw = ROBOT.split_props(s)
            hp = hq[..., None, :].expand(*pq.shape)
            pq_local = quat_canonical(quat_mul(quat_apply_inverse_quat(hp), quat_normalize(pq)))
            blocks = [quat_apply_inverse(hp, pp - p[..., None, :]), rot6d_from_quat(pq_local), quat_apply_inverse(hp, pv), pw]
            parts.append(torch.cat(blocks, dim=-1).flatten(-2))
        return torch.cat(parts, dim=-1) / self.in_scale

    def normalize_xi(self, xi: torch.Tensor) -> torch.Tensor:
        return (xi - self.xi_offset) / self.xi_scale

    # ------------------------------------------------------------------ integration

    def integrate(self, s: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Apply the (unscaled) network output ``delta`` (…, 70) to state ``s`` (…, 71)."""
        d = delta * self.out_scale
        n = NJ
        dv_h, dw_b, dqd = d[..., 0:3], d[..., 3:6], d[..., 6 : 6 + n]
        dp_h, dth_b, dq = d[..., 6 + n : 9 + n], d[..., 9 + n : 12 + n], d[..., 12 + n : 12 + 2 * n]
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        hq = heading_quat(q)
        v_new = v + quat_apply(hq, dv_h)
        w_new = w + dw_b
        qd_new = qd + dqd
        dt = self.dt if self.cfg.explicit_integration else 0.0
        p_new = p + dt * v_new + quat_apply(hq, dp_h)
        q_new = quat_normalize(quat_mul(q, quat_from_rotvec(dt * w_new + dth_b)))
        qj_new = qj + dt * qd_new + dq
        out = [p_new, q_new, qj_new, v_new, w_new, qd_new]
        if NP:
            pd = d[..., 12 + 2 * n :].reshape(*d.shape[:-1], NP, 12)
            pp, pq, pv, pw = ROBOT.split_props(s)
            hp = hq[..., None, :].expand(*pq.shape)
            pv_new = pv + quat_apply(hp, pd[..., 0:3])
            pw_new = pw + pd[..., 3:6]
            pp_new = pp + dt * pv_new + quat_apply(hp, pd[..., 6:9])
            pq_new = quat_normalize(quat_mul(quat_normalize(pq), quat_from_rotvec(dt * pw_new + pd[..., 9:12])))
            out.append(torch.cat([pp_new, pq_new, pv_new, pw_new], dim=-1).flatten(-2))
        return torch.cat(out, dim=-1)

    # ------------------------------------------------------------------ errors

    def error_vector(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Normalised 70-d error [dp, rotvec, dv, dw, dq, dqd] / sigma."""
        p1, q1, qj1, v1, w1, qd1 = self.split(s_pred)
        p2, q2, qj2, v2, w2, qd2 = self.split(s_true)
        q1 = quat_normalize(q1)
        q2 = quat_normalize(q2)
        rot_err = rotvec_from_quat(quat_mul(quat_apply_inverse_quat(q2), q1))
        parts = [p1 - p2, rot_err, v1 - v2, w1 - w2, qj1 - qj2, qd1 - qd2]
        if NP:
            a, b = ROBOT.split_props(s_pred), ROBOT.split_props(s_true)
            prot = rotvec_from_quat(quat_mul(quat_apply_inverse_quat(quat_normalize(b[1])), quat_normalize(a[1])))
            parts.append(torch.cat([a[0] - b[0], prot, a[2] - b[2], a[3] - b[3]], dim=-1).flatten(-2))
        return torch.cat(parts, dim=-1) / self.err_scale

    def normalized_sq_error(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Mean over dims of the squared normalised error (the reported 'prediction error')."""
        e = self.error_vector(s_pred, s_true)
        return (e * e).mean(-1)


def quat_apply_inverse_quat(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a unit quaternion (inverse)."""
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=q.dtype, device=q.device)

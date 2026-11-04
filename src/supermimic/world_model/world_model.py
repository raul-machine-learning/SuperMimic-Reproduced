"""World model: learned dynamics + learned termination head (+ learned grip force head) + analytic observation/reward heads."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from supermimic.robots import ROBOT
from supermimic.world_model.dynamics import MLP, DynamicsMLP
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg


@dataclass
class WorldModelCfg:
    hidden: tuple[int, ...] = (2048, 2048, 2048)
    term_hidden: tuple[int, ...] = (64, 64)
    state_spec: StateSpecCfg = field(default_factory=StateSpecCfg)
    term_loss_weight: float = 1.0
    # Scales of the two termination features (AnalyticHeads.termination_features).
    height_threshold: float = ROBOT.termination_thresholds[0]
    tilt_threshold: float = ROBOT.termination_thresholds[1]
    # Grip force head (robots with a grip): hidden sizes and loss weight.
    force_hidden: tuple[int, ...] = (512, 512)
    force_loss_weight: float = 1.0


class TerminationHead(nn.Module):
    """P(terminal | state, reference). Input: analytic termination features and their magnitudes."""

    def __init__(self, hidden=(64, 64), height_thr: float = 0.25, tilt_thr: float = 0.8):
        super().__init__()
        self.mlp = MLP(4, tuple(hidden), 1)
        self.register_buffer("scale", torch.tensor([1.0 / height_thr, 1.0 / tilt_thr]))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats (…, 2) (e.g. [height error, tilt error]) -> logits (…,)"""
        x = feats * self.scale
        return self.mlp(torch.cat([x, x.abs()], dim=-1)).squeeze(-1)


class ForceHead(nn.Module):
    """Contact normal force (N) of each gripping hand's force bodies on its prop in a state: from the state's
    canonical features, the set-point that drives the fingers (position servos push in proportion to how far it
    lies inside the prop) and the taxels' signed distances to the prop (``AnalyticHeads.grip_distances``)."""

    D_CLIP = (-0.01, 0.03)  # m
    D_SCALE = 0.01  # m

    def __init__(self, feat_dim: int, act_dim: int, hands: int, bodies: int, hidden=(512, 512)):
        super().__init__()
        from supermimic.kinematics.glove import NTAXEL

        self.hands, self.bodies = hands, bodies
        self.mlp = MLP(feat_dim + act_dim + 2 * hands * NTAXEL, tuple(hidden), hands * bodies)

    def forward(self, feat: torch.Tensor, action: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """feat (…, FEAT), action (…, A), d (…, G, 129) -> (…, G, B) forces >= 0."""
        from supermimic.kinematics.glove import GloveModel

        x = torch.cat([feat, action, (d.clamp(*self.D_CLIP) / self.D_SCALE).flatten(-2), GloveModel.touching(d).flatten(-2)], dim=-1)
        return F.softplus(self.mlp(x)).unflatten(-1, (self.hands, self.bodies))


def force_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """(…,) squared error of log(1 + F / 1 N), mean over hands and bodies: contacts range from grazing to tens of N."""
    return ((torch.log1p(pred) - torch.log1p(true)) ** 2).mean((-2, -1))


class WorldModel(nn.Module):
    def __init__(self, cfg: WorldModelCfg, heads: AnalyticHeads, device):
        super().__init__()
        self.cfg = cfg
        self.spec = StateSpec(cfg.state_spec, device)
        self.dynamics = DynamicsMLP(hidden=cfg.hidden)
        self.term_head = TerminationHead(cfg.term_hidden, cfg.height_threshold, cfg.tilt_threshold)
        self.force_head = None
        if ROBOT.grip:
            from supermimic.envs.mdp.grip_force import num_bodies

            self.force_head = ForceHead(ROBOT.feat_dim, ROBOT.na, len(ROBOT.grip), num_bodies(), cfg.force_hidden)
        self.heads = heads  # analytic observation / reward / FK (no parameters)
        self.to(device)

    # ------------------------------------------------------------------ one step

    def step(self, s: torch.Tensor, action: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """s_{t+1} = integrate(s_t, f(s_t, a_t, xi))."""
        feat = self.spec.canonical(s)
        delta = self.dynamics(feat, action, self.spec.normalize_xi(xi))
        return self.spec.integrate(s, delta)

    def grip_forces(self, s_next: torch.Tensor, action: torch.Tensor, bodies=None) -> tuple[torch.Tensor, torch.Tensor]:
        """(taxel distances (…, G, 129), predicted force-body forces (…, G, B)) of s_next reached under the set-point
        ``action`` (the dynamics' action input)."""
        assert self.force_head is not None
        b = bodies if bodies is not None else self.heads.robot_bodies(s_next)
        d = self.heads.grip_distances(s_next, b)
        return d, self.force_head(self.spec.canonical(s_next), action, d)

    def termination_logits(self, s_next: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        # Termination features read the root pose or the wrist joints straight from the state: no FK needed.
        feats = self.heads.termination_features(s_next, ref)
        return self.term_head(feats)

    # ------------------------------------------------------------------ Eq. 1

    def _loss_step(self, s, a, xi, s_true, done, ref_root_pos, ref_root_quat, ref_joint_pos, ref_prop_pos=None, f_true=None):
        """One autoregressive step of Eq. 1: returns (s_next, sq_err (n,70), term_bce (n,), logits (n,), force loss (n,)
        and absolute force error (n,) in N, or None without ``f_true`` (n, G*B))."""
        s_next = self.step(s, a, xi)
        e = self.spec.error_vector(s_next, s_true)
        feats = self.heads.termination_features(s_next, ReferenceFrame(None, None, None, None, ref_root_pos, ref_root_quat, ref_joint_pos, ref_prop_pos))
        logits = self.term_head(feats)
        bce = F.binary_cross_entropy_with_logits(logits, done, reduction="none")
        fl = f_err = None
        if f_true is not None:
            # The force labels train the head only: a millimetre-sensitive contact loss must not reshape the dynamics.
            pred = self.grip_forces(s_next.detach(), a)[1]
            true = f_true.view_as(pred)
            fl = force_loss(pred, true)
            f_err = (pred - true).abs().mean((-2, -1))
        return s_next, e * e, bce, logits, fl, f_err

    def enable_compile(self, mode: str | None = "default") -> None:
        self._loss_step_fn = torch.compile(self._loss_step, mode=None if mode == "default" else mode, dynamic=False) if mode else self._loss_step

    def window_loss(self, states: torch.Tensor, a_exec: torch.Tensor, xi: torch.Tensor, done: torch.Tensor, refs: list[ReferenceFrame],
                    mask: torch.Tensor | None = None, forces: torch.Tensor | None = None):
        """k-step autoregressive loss.

        states: (k+1, n, 71) logged states; a_exec: (k, n, 29); xi: (n, 33);
        done: (k, n) termination label of states 1..k; refs[j]: reference of transition j (for the
        termination features of state j+1); mask: (k, n) 0 for the padding of windows that end early;
        forces: (k+1, n, G*B) grip contact forces measured in the logged states (fits the force head).
        """
        k = a_exec.shape[0]
        s = states[0]
        total_state = 0.0
        total_term = 0.0
        metrics = {}
        err_first = None
        err_last = None
        acc_num = 0.0
        total_force = 0.0
        force_mae = 0.0
        use_force = forces is not None and self.force_head is not None
        step_fn = getattr(self, "_loss_step_fn", None) or self._loss_step
        for j in range(k):
            s, sq, bce, logits, fl, f_err = step_fn(s, a_exec[j], xi, states[j + 1], done[j], refs[j].root_pos_w, refs[j].root_quat_w, refs[j].joint_pos,
                                                    refs[j].prop_pos_w, forces[j + 1] if use_force else None)
            if mask is None:
                mean = lambda x: x.mean()  # noqa: E731
            else:
                m = mask[j]
                mean = lambda x, m=m: torch.where(m > 0, x, 0.0).sum() / m.sum().clamp_min(1.0)  # noqa: E731
            total_state = total_state + mean(sq.sum(-1))
            total_term = total_term + mean(bce)
            if use_force:
                total_force = total_force + mean(fl)
                force_mae = force_mae + mean(f_err).detach()
            with torch.no_grad():
                nse = mean(sq.mean(-1))
                if j == 0:
                    err_first = nse
                err_last = nse
                acc_num += mean(((logits > 0).float() == done[j]).float())
        loss = total_state / k + self.cfg.term_loss_weight * total_term / k
        if use_force:
            loss = loss + self.cfg.force_loss_weight * total_force / k
            metrics.update(force_loss=(total_force / k).detach(), force_mae_n=force_mae / k)
        # Metrics stay on-device (detached tensors): a float() here would stall the CPU on the
        # forward pass every minibatch, which in data-parallel runs turns into lockstep waiting.
        metrics.update(
            state_loss=(total_state / k).detach(),
            term_bce=(total_term / k).detach(),
            nse_1step=err_first.detach(),
            nse_kstep=err_last.detach(),
            term_acc=(acc_num / k).detach(),
        )
        return loss, metrics

    # ------------------------------------------------------------------ diagnostics

    @torch.no_grad()
    def rollout_error(self, states: torch.Tensor, a_exec: torch.Tensor, xi: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Open-loop k-step normalised squared error per step (k,), averaged over the batch (real transitions only)."""
        k = a_exec.shape[0]
        s = states[0]
        out = []
        for j in range(k):
            s = self.step(s, a_exec[j], xi)
            e = self.spec.normalized_sq_error(s, states[j + 1])
            out.append(e.mean() if mask is None else torch.where(mask[j] > 0, e, 0.0).sum() / mask[j].sum().clamp_min(1.0))
        return torch.stack(out)

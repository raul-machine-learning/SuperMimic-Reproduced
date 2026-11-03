"""Read / write the full simulator state as one tensor (71-d for the G1).

Layout (``ROBOT.state_dim``): robot ``[qpos | qvel]``, then per prop of ``ROBOT.props`` its free-joint
``[qpos 7 | qvel 6]``, gathered from the scene entities' addresses in MuJoCo's qpos / qvel.
"""

from __future__ import annotations

import torch

from supermimic.robots import ROBOT

STATE_DIM = ROBOT.state_dim
NQ = ROBOT.nq
NV = ROBOT.nv


class StateIO:
    def __init__(self, env):
        self.env = env
        self.sim = env.sim
        # States are stored relative to each env's origin so that they are directly comparable
        # with the (origin-free) motion library and translation-invariant across envs.
        self.origins = env.scene.env_origins.clone()  # (N, 3)
        nq, nv = self.sim.data.qpos.shape[-1], self.sim.data.qvel.shape[-1]
        assert nq == NQ + 7 * ROBOT.num_props and nv == NV + 6 * ROBOT.num_props, (nq, nv)
        robot = env.scene["robot"].indexing
        q_adr = [robot.free_joint_q_adr, robot.joint_q_adr]
        v_adr = [robot.free_joint_v_adr, robot.joint_v_adr]
        for p in ROBOT.props:
            ix = env.scene[p].indexing
            q_adr.append(ix.free_joint_q_adr)
            v_adr.append(ix.free_joint_v_adr)
        dev = self.origins.device
        q_adr = [a.to(device=dev, dtype=torch.long) for a in q_adr]
        v_adr = [a.to(device=dev, dtype=torch.long) for a in v_adr]
        # Gather index into cat([qpos, qvel]) for each state column.
        cols = [q_adr[0], q_adr[1], nq + v_adr[0], nq + v_adr[1]]
        for k in range(ROBOT.num_props):
            cols += [q_adr[2 + k], nq + v_adr[2 + k]]
        self.cols = torch.cat(cols)
        assert self.cols.numel() == STATE_DIM and torch.unique(self.cols).numel() == STATE_DIM
        self.identity = bool(torch.equal(self.cols, torch.arange(STATE_DIM, device=dev)))
        self.nq = nq
        # State columns holding a world position: the robot root and every prop.
        r = ROBOT.robot_state_dim
        self.pos_cols = [slice(0, 3)] + [slice(r + 13 * k, r + 13 * k + 3) for k in range(ROBOT.num_props)]

    def read(self) -> torch.Tensor:
        """(N, STATE_DIM) copy of the state."""
        # Index with [:] to get plain tensors out of mjlab's TorchArray proxies (torch.cat on the
        # proxies recurses inside their __torch_function__).
        s = torch.cat([self.sim.data.qpos[:], self.sim.data.qvel[:]], dim=-1)
        s = s.clone() if self.identity else s[:, self.cols]
        for c in self.pos_cols:
            s[:, c] -= self.origins
        return s

    @torch.no_grad()
    def write(self, env_ids: torch.Tensor, state: torch.Tensor, forward: bool = True) -> None:
        """Overwrite qpos/qvel of ``env_ids`` and refresh derived quantities."""
        state = state.to(self.sim.data.qpos.dtype).clone()
        for c in self.pos_cols:
            state[:, c] += self.origins[env_ids]
        if self.identity:
            self.sim.data.qpos[env_ids] = state[:, :NQ]
            self.sim.data.qvel[env_ids] = state[:, NQ:]
        else:
            full = torch.cat([self.sim.data.qpos[env_ids], self.sim.data.qvel[env_ids]], dim=-1)
            full[:, self.cols] = state
            self.sim.data.qpos[env_ids] = full[:, : self.nq]
            self.sim.data.qvel[env_ids] = full[:, self.nq :]
        if hasattr(self.sim.data, "qacc_warmstart"):
            self.sim.data.qacc_warmstart[env_ids] = 0.0
        if forward:
            self.sim.forward()

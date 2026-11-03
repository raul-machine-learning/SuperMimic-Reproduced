"""Differentiable batched forward kinematics for a floating-base MuJoCo tree (torch).

Built from a compiled ``mujoco.MjModel`` (the G1, G1 + Inspire hands, the capsule-hand rig) so the tree, joint
axes, anchors and ``qpos0`` are exactly MuJoCo's. Reproduces, for every body:

* ``xpos`` / ``xquat``                      (link frame origin pose, world)
* link-frame-origin linear velocity, world  (== mjlab ``body_link_lin_vel_w``)
* angular velocity, world                   (== mjlab ``body_link_ang_vel_w``)

given ``qpos`` (root pos, root quat wxyz, then the joints) and ``qvel`` (root linear velocity in world, root
angular velocity in **body** frame, then the joint rates), i.e. MuJoCo's free-joint conventions.

Quaternion math is element-wise (no 3x3 matmuls) so results are unaffected by TF32.
Position FK follows ``mj_kinematics``: a body's joints act in order, a hinge rotating about its anchor by
``qpos - qpos0``, a slide translating along its axis by ``qpos - qpos0``. Velocities are propagated down the tree:

    w_b = w_parent + sum_hinges axis_w * qdot
    v_b = v_parent + w_parent x (p_b - p_parent) + sum_hinges (axis_w * qdot) x (p_b - anchor_w) + sum_slides axis_w * qdot

which equals MuJoCo's ``cvel`` transported to the body origin (``mj_objectVelocity`` with ``flg_local=0``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch

from supermimic.kinematics.rotations import quat_apply, quat_from_angle_axis, quat_mul, quat_normalize

_HINGE = mujoco.mjtJoint.mjJNT_HINGE
_SLIDE = mujoco.mjtJoint.mjJNT_SLIDE
_FREE = mujoco.mjtJoint.mjJNT_FREE


@dataclass
class _Joint:
    joint_type: int  # mjJNT_HINGE / mjJNT_SLIDE
    qpos_adr: int
    dof_adr: int
    pos: torch.Tensor  # (3,) anchor in the body frame
    axis: torch.Tensor  # (3,) in the body frame
    qpos0: float


@dataclass
class _BodyLink:
    body_id: int  # MuJoCo body id (1..nbody-1)
    parent_index: int  # index into our body list (0..nb-1), -1 for the root
    free: bool = False
    joints: list[_Joint] = field(default_factory=list)


class G1Kinematics:
    """Batched differentiable FK for a MuJoCo single-tree model with a free root."""

    def __init__(
        self,
        model: mujoco.MjModel,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.nbody = model.nbody - 1  # exclude world
        self.nq = model.nq
        self.nv = model.nv
        self.body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]
        self.joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=self.device)  # noqa: E731

        links: list[_BodyLink] = []
        for b in range(1, model.nbody):
            parent = int(model.body_parentid[b])
            assert parent < b, "MuJoCo bodies are topologically ordered"
            link = _BodyLink(b, -1 if parent == 0 else parent - 1)
            for j in range(int(model.body_jntadr[b]), int(model.body_jntadr[b]) + int(model.body_jntnum[b])):
                jt = int(model.jnt_type[j])
                if jt == _FREE:
                    assert b == 1 and model.body_jntnum[b] == 1, "only the root body may carry the free joint"
                    link.free = True
                    continue
                assert jt in (_HINGE, _SLIDE), f"joint type {jt}"
                link.joints.append(_Joint(jt, int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]), t(model.jnt_pos[j]),
                                          t(model.jnt_axis[j]), float(model.qpos0[model.jnt_qposadr[j]])))
            links.append(link)
        assert links[0].free and links[0].parent_index == -1
        self.links = links
        self.body_pos = t(model.body_pos[1:])  # (nb,3)
        self.body_quat = t(model.body_quat[1:])  # (nb,4) wxyz

    def body_index(self, name: str) -> int:
        return self.body_names.index(name)

    def body_indices(self, names) -> torch.Tensor:
        return torch.tensor([self.body_names.index(n) for n in names], device=self.device)

    # ------------------------------------------------------------------ core

    def forward(self, qpos: torch.Tensor, qvel: torch.Tensor | None = None):
        """FK for a batch.

        Args:
          qpos: (..., nq) MuJoCo generalized positions.
          qvel: (..., nv) MuJoCo generalized velocities (free joint: lin world, ang body).
        Returns:
          dict with 'pos' (..., nb, 3), 'quat' (..., nb, 4) and, if qvel is given,
          'lin_vel' (..., nb, 3), 'ang_vel' (..., nb, 3) in world frame at the link origin.
        """
        batch = qpos.shape[:-1]
        nb = self.nbody
        pos = [None] * nb
        quat = [None] * nb
        lin = [None] * nb
        ang = [None] * nb
        compute_vel = qvel is not None

        # Root (free joint).
        root_pos = qpos[..., 0:3]
        root_quat = quat_normalize(qpos[..., 3:7])
        pos[0] = root_pos
        quat[0] = root_quat
        if compute_vel:
            lin[0] = qvel[..., 0:3]
            ang[0] = quat_apply(root_quat, qvel[..., 3:6])  # body -> world

        for i in range(1, nb):
            l = self.links[i]
            p = l.parent_index
            pq = quat[p]
            pp = pos[p]
            # Body frame from parent frame (fixed offset), then the body's joints in order.
            b_pos = pp + quat_apply(pq, self.body_pos[i].expand(*batch, 3))
            b_quat = quat_mul(pq, self.body_quat[i].expand(*batch, 4))
            twists = []  # (angular rate, anchor) of hinges, linear rate of slides
            for jt in l.joints:
                axis_w = quat_apply(b_quat, jt.axis.expand(*batch, 3))
                dq = qpos[..., jt.qpos_adr] - jt.qpos0
                qd = qvel[..., jt.dof_adr].unsqueeze(-1) if compute_vel else None
                if jt.joint_type == _HINGE:
                    anchor = b_pos + quat_apply(b_quat, jt.pos.expand(*batch, 3))
                    b_quat = quat_mul(b_quat, quat_from_angle_axis(dq, jt.axis.expand(*batch, 3)))
                    b_pos = anchor - quat_apply(b_quat, jt.pos.expand(*batch, 3))
                    if compute_vel:
                        twists.append((axis_w * qd, anchor))
                else:
                    b_pos = b_pos + axis_w * dq.unsqueeze(-1)
                    if compute_vel:
                        twists.append((None, axis_w * qd))
            if compute_vel:
                w_b = ang[p]
                v_b = lin[p] + torch.cross(ang[p], b_pos - pp, dim=-1)
                for w_j, x in twists:
                    if w_j is None:
                        v_b = v_b + x
                    else:
                        w_b = w_b + w_j
                        v_b = v_b + torch.cross(w_j, b_pos - x, dim=-1)
                ang[i] = w_b
                lin[i] = v_b
            pos[i] = b_pos
            quat[i] = b_quat

        out = {"pos": torch.stack(pos, dim=-2), "quat": torch.stack(quat, dim=-2)}
        if compute_vel:
            out["lin_vel"] = torch.stack(lin, dim=-2)
            out["ang_vel"] = torch.stack(ang, dim=-2)
        return out

    __call__ = forward

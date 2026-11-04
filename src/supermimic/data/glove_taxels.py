"""Glove taxels placed on the capsule hands (scripts/assets/build_capsule_hands.py).

The recording does not ship the taxel positions (config.json lists ``hand_pressure_pos.h5``, which is absent), so
each of the 129 taxels per hand is placed from the glove numbering in ``supermimic.data.sweep_mocap`` on the
palm-side surface of the segment under it (a capsule, or a box with ``SM_HAND_SHAPE=box``), with the outward
surface normal:

  fingertip pads   on the distal capsule: rows from the tip toward the base, columns from the thumb side to the
                   little-finger side (index-finger pads 3 x 3, thumb pad 4 x 3)
  middle, proximal one taxel at the middle of that phalanx
  thumb base       on the thumb metacarpal, turned 60 deg from its palm side toward the outside of the hand
  palm rows        five rows between the finger bases and the heel of the hand, each running from the index side
                   to the little-finger side across the palm rays, projected onto the palm capsules along the palm
                   normal

On a box, angles around the bone count from the box face nearest the segment's palm side, and a point is placed
where its direction leaves the square cross-section: a pad spans that face, the thumb base sits on the side face;
the tip rows stay on the face, which runs one half-width past the joint.

The ordering inside a pad and the thumb-base side are assumptions (the glove documentation gives only the counts);
every region keeps its taxels, so region-level loads are unaffected by them.

The taxels come in blocks (``block_bounds``): each fingertip pad and the palm is an array with a center and a scale
of its spacing, each single taxel has a center. ``taxel_layout_torch`` builds the layout differentiably in those
parameters; scripts/data/fit_glove_blocks.py fits them per hand to the glove's pressure at the mocap pose
(``fitted_blocks``, used by ``glove_layout``). Without that fit, the shifts and rolls of the older on/off fit
(scripts/data/fit_glove_taxels.py, ``fitted_adjust``) are used.

The palm side comes from anatomy alone (the thumb is on the index side; the left hand mirrors the right), not from
the hinge axes, so ``flexion_check`` can confirm that each finger flexes toward it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch

from supermimic.robots import HAND_ASSETS

FIT_FILE = HAND_ASSETS / "glove_taxel_fit.json"
BLOCK_FIT_FILE = HAND_ASSETS / "glove_block_fit.json"

FINGERS4 = ("index", "middle", "ring", "pinky")
GLOVE_BASE = {"index": 13, "middle": 24, "ring": 35, "pinky": 46}
PALM_ROWS = ((57, 69), (69, 84), (84, 99), (99, 114), (114, 129))
PALM_ROW_FRACTION = (0.90, 0.70, 0.50, 0.30, 0.08)  # along each palm ray, heel -> finger base
PAD_SPREAD = np.radians(35.0)
THUMB_BASE_TURN = np.radians(60.0)


@dataclass
class TaxelLayout:
    body: np.ndarray  # (129,) body id
    pos: np.ndarray  # (129, 3) in the body frame
    normal: np.ndarray  # (129, 3) outward unit normal in the body frame


@dataclass
class Segment:
    """A hand segment in world: the bone from ``a`` (proximal) to ``b`` and its half-width ``r``; a capsule of radius
    ``r`` around it, or (``axes`` set: (3, 3) columns = the box's x, y, z = bone axes) a box reaching ``r`` past
    both ends."""

    a: np.ndarray
    b: np.ndarray
    r: float
    axes: np.ndarray | None = None

    def __iter__(self):
        return iter((self.a, self.b, self.r))


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def palm_frame(model: mujoco.MjModel, data: mujoco.MjData, S: str):
    """(wrist, palm normal, radial direction, finger-base positions) in world, after kinematics. The palm normal
    points out of the palm; radial points from the little-finger side to the index side."""
    wrist = data.xpos[model.body(f"{S}_hand").id].copy()
    mcp = {f: data.xpos[model.body(f"{S}_{f}1").id].copy() for f in FINGERS4}
    along = _unit(mcp["middle"] - wrist)
    radial = _unit(mcp["index"] - mcp["pinky"])
    # Right hand: palm normal = radial x along; the left hand is the mirror image.
    n = _unit(np.cross(radial, along) if S == "R" else np.cross(along, radial))
    return wrist, n, _unit(radial - (radial @ n) * n), mcp


def flexion_alignment(model: mujoco.MjModel, data: mujoco.MjData, joint: str) -> float:
    """Cosine between the fingertip's velocity under positive rotation of ``joint`` and the direction from the tip
    to a point 4 cm in front of the middle of the palm (positive: the joint closes the hand), after kinematics."""
    S, rest = joint.split("_", 1)
    finger = rest[: next(i for i, ch in enumerate(rest) if ch.isdigit())]
    wrist, n, _, mcp = palm_frame(model, data, S)
    target = wrist + 0.5 * (mcp["middle"] - wrist) + 0.04 * n
    j = model.joint(joint).id
    tip = data.site_xpos[model.site(f"{S}_{finger}_tip").id]
    v = np.cross(data.xaxis[j], tip - data.xanchor[j])
    return float(_unit(v) @ _unit(target - tip))


class _Pose:
    """World geometry of one hand at the model's zero pose."""

    def __init__(self, model: mujoco.MjModel, S: str):
        self.m, self.S = model, S
        self.d = mujoco.MjData(model)
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self.d.qpos[model.jnt_qposadr[j] + 3] = 1.0
        mujoco.mj_kinematics(model, self.d)
        self.wrist, self.palm_n, self.radial, self.mcp = palm_frame(model, self.d, S)
        self.heel = {f: self.capsule("hand", f"{f}_wrist").a for f in FINGERS4}

    def bid(self, short: str) -> int:
        return self.m.body(f"{self.S}_{short}").id

    def capsule(self, short: str, geom: str | None = None) -> Segment:
        """The segment of a capsule or box geom in world, proximal = the end nearer the body origin."""
        g = self.m.geom(f"{self.S}_{geom or short}_collision").id
        R, size = self.d.geom_xmat[g].reshape(3, 3), self.m.geom_size[g]
        box = self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
        r, h = (size[0], size[2] - size[0]) if box else (size[0], size[1])
        c, z = self.d.geom_xpos[g], R[:, 2]
        a, b = c - h * z, c + h * z
        o = self.d.xpos[self.m.geom_bodyid[g]]
        if np.linalg.norm(a - o) > np.linalg.norm(b - o):
            a, b = b, a
        return Segment(a, b, r, R.copy() if box else None)

    def flex_axis(self, short: str) -> np.ndarray:
        return self.d.xaxis[self.m.joint(f"{self.S}_{short}_flex").id].copy()

    def segment_palm_side(self, short: str) -> np.ndarray:
        """Direction a point on the segment moves under positive flexion of the segment's own hinge."""
        a, b, _ = self.capsule(short)
        return _unit(np.cross(self.flex_axis(short), b - a))


_POSES: dict[tuple[int, str], _Pose] = {}


def _surface(seg: Segment, n: np.ndarray, lat: np.ndarray, axial, arc):
    """Points and normals (torch, ..., 3) on a segment: ``axial`` m from the proximal end, ``arc`` m around the bone
    from its palm side ``n`` toward ``lat`` (arc length on the cross-section's perimeter; both torch tensors). On a
    capsule, past the distal end wraps onto its cap. On a box, the palm side is the face nearest ``n``; past its edge
    the arc runs on to the side face; ``axial`` stays within the box's span."""
    a, b, r = seg
    u = _unit(b - a)
    n = _unit(n - (n @ u) * u)
    lat = _unit(lat - (lat @ u) * u - (lat @ n) * n)
    L = float(np.linalg.norm(b - a))
    T = lambda v: torch.as_tensor(v, dtype=torch.float64)
    if seg.axes is None:
        phi = arc / r
        over = (axial - L).clamp(min=0.0) / r
        d = torch.cos(phi)[..., None] * T(n) + torch.sin(phi)[..., None] * T(lat) + over[..., None] * T(u)
        d = d / d.norm(dim=-1, keepdim=True)
        return T(a) + axial.clamp(0.0, L)[..., None] * T(u) + r * d, d
    X, Y = seg.axes[:, 0], seg.axes[:, 1]
    face = max((X, -X, Y, -Y), key=lambda f: f @ n)
    lat = _unit(lat - (lat @ face) * face - (lat @ u) * u)
    t = (arc / r).clamp(-3.0, 3.0)
    on_face = t.abs() <= 1.0
    along_face = torch.where(on_face, torch.ones_like(t), 2.0 - t.abs())
    across = torch.where(on_face, t, torch.sign(t))
    p = T(a) + axial.clamp(-r, L + r)[..., None] * T(u) + r * (along_face[..., None] * T(face) + across[..., None] * T(lat))
    nrm = torch.where(on_face[..., None], T(face).expand(p.shape), torch.sign(t)[..., None] * T(lat))
    return p, nrm


def _capsule_gap(pts: np.ndarray, segs) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance (...,) of points (..., 3) outside the union of the segments (capsules or boxes), and the
    segment whose surface is nearest each point."""
    gaps = []
    for seg in segs:
        a, b, r = seg
        if seg.axes is None:
            ab = b - a
            s = np.clip((pts - a) @ ab / (ab @ ab), 0.0, 1.0)
            gaps.append(np.linalg.norm(pts - (a + s[..., None] * ab), axis=-1) - r)
        else:
            half = np.array([r, r, 0.5 * np.linalg.norm(b - a) + r])
            q = np.abs((pts - 0.5 * (a + b)) @ seg.axes) - half
            gaps.append(np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(-1), 0.0))
    gaps = np.stack(gaps, -1)
    return gaps.min(-1), np.abs(gaps).argmin(-1)


def _palm_points(pose: _Pose, base: np.ndarray, palm_caps) -> tuple[np.ndarray, np.ndarray]:
    """Exit points (N, 3) of the rays base + t * palm normal through the union of the palm segments (bridging the
    few-mm gaps between neighbouring metacarpals, as the glove's palm does), and the segment each lands on."""
    ts = np.linspace(0.0, 0.06, 301)
    pts = base[:, None] + ts[None, :, None] * pose.palm_n
    gap, _ = _capsule_gap(pts, palm_caps)
    out = np.zeros_like(base)
    for i in range(len(base)):
        for tol in (0.0, 0.002, 0.004, 0.008):
            hit = np.nonzero(gap[i] <= tol)[0]
            if len(hit):
                out[i] = pts[i, hit.max()]
                break
        else:
            raise ValueError("palm ray misses every palm capsule")
    return out, _capsule_gap(out, palm_caps)[1]


PAD_BLOCKS = {"thumb_pad": (0, "thumb3", 4, 3), **{f"{f}_pad": (b0, f"{f}3", 3, 3) for f, b0 in GLOVE_BASE.items()}}
SINGLE_BLOCKS = {"thumb_base": (12, "thumb1", THUMB_BASE_TURN),
                 **{f"{f}_{k}": (b0 + o, f"{f}{j}", 0.0) for f, b0 in GLOVE_BASE.items() for k, o, j in (("middle", 9, 2), ("proximal", 10, 1))}}


def block_bounds() -> dict[str, tuple[tuple[float, float], ...]]:
    """The parameters of each taxel block, with their bounds. Arrays (fingertip pads, palm): (center shift along the
    segment toward its tip / toward the finger bases (m), center shift toward the thumb / index side (m, around the
    segment for a pad), scale of the array's spacing); single taxels: the first two."""
    shift = (-0.010, 0.010)
    out: dict[str, tuple[tuple[float, float], ...]] = {k: (shift, shift, (0.6, 1.6)) for k in PAD_BLOCKS}
    out.update({k: (shift, shift) for k in SINGLE_BLOCKS})
    out["palm"] = ((-0.020, 0.020), (-0.020, 0.020), (0.6, 1.5))
    return out


def neutral_blocks() -> dict[str, tuple[float, ...]]:
    return {k: (0.0, 0.0, 1.0)[: len(v)] for k, v in block_bounds().items()}


def adjust_groups() -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    """Layout adjustments of scripts/data/fit_glove_taxels.py, with their bounds. Pads and single taxels: (shift along
    the segment toward its tip (m), roll about the segment toward the thumb side (rad)); palm: (shift toward the
    finger bases (m), shift toward the index side (m)). ``taxel_layout`` reads two-value pad and single entries as
    these (scale 1, roll = arc / radius)."""
    seg = ((-0.008, 0.008), (-1.0, 1.0))
    out = {"thumb_pad": seg, "thumb_base": seg}
    for f in FINGERS4:
        out[f"{f}_pad"], out[f"{f}_middle"], out[f"{f}_proximal"] = seg, seg, seg
    out["palm"] = ((-0.015, 0.015), (-0.012, 0.012))
    return out


def fitted_adjust() -> dict[str, tuple[float, float]]:
    """The adjustments fitted to the glove data by scripts/data/fit_glove_taxels.py (none if it has not run)."""
    return {k: tuple(v) for k, v in json.loads(FIT_FILE.read_text())["adjust"].items()} if FIT_FILE.exists() else {}


def fitted_blocks(S: str) -> dict[str, tuple[float, ...]]:
    """Hand ``S``'s block parameters fitted by scripts/data/fit_glove_blocks.py (none if it has not run)."""
    if not BLOCK_FIT_FILE.exists():
        return {}
    return {k: tuple(v) for k, v in json.loads(BLOCK_FIT_FILE.read_text())["blocks"][S].items()}


def glove_layout(model: mujoco.MjModel, S: str) -> TaxelLayout:
    """The layout of the glove reward and the renders: ``fitted_blocks`` when fitted, else ``fitted_adjust``."""
    if BLOCK_FIT_FILE.exists():
        return taxel_layout(model, S, fitted_blocks(S))
    return taxel_layout(model, S, fitted_adjust(), roll=True)


def _pose(model: mujoco.MjModel, S: str) -> _Pose:
    pose = _POSES.get((id(model), S))
    if pose is None or pose.m is not model:
        pose = _POSES[(id(model), S)] = _Pose(model, S)
    return pose


def taxel_layout_torch(model: mujoco.MjModel, S: str, blocks: dict[str, torch.Tensor | tuple[float, ...]] | None = None,
                       roll: bool = False) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """(body ids (129,), positions (129, 3) and normals (129, 3) in the body frames, float64), differentiable in the
    block parameters (``block_bounds``; missing blocks: ``neutral_blocks``). The palm points are where rays along the
    palm normal leave the palm segments: their height along the normal is not differentiated (zero on flat box faces),
    their position in the palm plane is. ``roll``: the pads' and single taxels' second value is a roll about the
    segment (rad, ``adjust_groups``) instead of an arc (m), and pads take no scale."""
    pose = _pose(model, S)
    T = lambda v: torch.as_tensor(v, dtype=torch.float64)
    blocks = blocks or {}
    body = np.zeros(129, int)
    pos_w: list = [None] * 129
    nrm_w: list = [None] * 129

    def get(name: str, n: int):
        v = blocks.get(name, neutral_blocks()[name])
        v = v if isinstance(v, torch.Tensor) else T(v)
        return v if len(v) >= n else torch.cat([v, T([1.0])])

    def put(ids, short: str, p, nrm) -> None:
        for k, i in enumerate(ids):
            body[i], pos_w[i], nrm_w[i] = pose.bid(short), p[k], nrm[k]

    for name, (first, short, rows, cols) in PAD_BLOCKS.items():
        seg = pose.capsule(short)
        L = float(np.linalg.norm(seg.b - seg.a))
        da, dl, k = get(name, 3)
        axial0 = T(np.linspace(L + 0.45 * seg.r, 0.0, rows))
        arc0 = T(seg.r * np.linspace(PAD_SPREAD, -PAD_SPREAD, cols))
        dl = dl * seg.r if roll else dl
        axial = axial0.mean() + da + k * (axial0 - axial0.mean())
        arc = dl + k * arc0
        p, nrm = _surface(seg, pose.segment_palm_side(short), pose.radial, axial[:, None].expand(rows, cols).reshape(-1),
                          arc[None, :].expand(rows, cols).reshape(-1))
        put(range(first, first + rows * cols), short, p, nrm)
    for name, (i, short, phi) in SINGLE_BLOCKS.items():
        seg = pose.capsule(short)
        da, dl = get(name, 2)[:2]
        dl = dl * seg.r if roll else dl
        p, nrm = _surface(seg, pose.segment_palm_side(short), pose.radial, (0.5 * float(np.linalg.norm(seg.b - seg.a)) + da)[None],
                          (seg.r * phi + dl)[None])
        put([i], short, p, nrm)

    p_along, p_lat, k = get("palm", 3)
    n = T(pose.palm_n)
    along = pose.mcp["middle"] - pose.heel["middle"]
    along = T(_unit(along - (along @ pose.palm_n) * pose.palm_n))
    palm_shorts = ([(f"{f}0", None) for f in FINGERS4] + [("hand", f"{f}_wrist") for f in FINGERS4]
                   + [("hand", "heel"), ("hand", "thenar")])
    palm_caps = [pose.capsule(s, g) for s, g in palm_shorts]
    ids, bases = [], []
    for (lo, hi), frac in zip(PALM_ROWS, PALM_ROW_FRACTION):
        line = [pose.heel[f] + frac * (pose.mcp[f] - pose.heel[f]) for f in FINGERS4]
        for i, s in enumerate(np.linspace(0.0, 3.0, hi - lo)):
            kk = min(int(s), 2)
            w = s - kk
            ids.append(lo + i)
            bases.append((1 - w) * line[kk] + w * line[kk + 1] - 0.03 * pose.palm_n)
    base0 = T(np.array(bases))
    rel = base0 - base0.mean(0)
    rel = rel - (rel @ n)[:, None] * n
    base = base0 + (k - 1.0) * rel + p_along * along + p_lat * T(pose.radial)
    pts, which = _palm_points(pose, base.detach().numpy(), palm_caps)
    height = T((pts - base.detach().numpy()) @ pose.palm_n)
    for j, (i, cap) in enumerate(zip(ids, which)):
        body[i], pos_w[i], nrm_w[i] = pose.bid(palm_shorts[int(cap)][0]), base[j] + height[j] * n, n

    pos_w, nrm_w = torch.stack(pos_w), torch.stack(nrm_w)
    R = T(pose.d.xmat[body].reshape(-1, 3, 3))
    pos = torch.einsum("nji,nj->ni", R, pos_w - T(pose.d.xpos[body]))
    nrm = torch.einsum("nji,nj->ni", R, nrm_w)
    return body, pos, nrm


def taxel_layout(model: mujoco.MjModel, S: str, adjust: dict[str, tuple[float, ...]] | None = None, roll: bool = False) -> TaxelLayout:
    """``adjust``: block -> the values of ``block_bounds`` (``roll``: of ``adjust_groups``); missing blocks:
    ``neutral_blocks``."""
    with torch.no_grad():
        body, pos, nrm = taxel_layout_torch(model, S, adjust, roll)
    return TaxelLayout(body=body, pos=pos.numpy(), normal=nrm.numpy())


def flexion_check(model: mujoco.MjModel, data: mujoco.MjData, S: str) -> dict[str, float]:
    """``flexion_alignment`` of every flexion hinge of one hand at the pose in ``data`` (after kinematics)."""
    return {model.joint(j).name: flexion_alignment(model, data, model.joint(j).name)
            for j in range(model.njnt) if model.joint(j).name.startswith(f"{S}_") and model.joint(j).name.endswith("_flex")}


def world_taxels(data: mujoco.MjData, layout: TaxelLayout) -> tuple[np.ndarray, np.ndarray]:
    """(129, 3) positions and normals in world for the current ``data`` (after kinematics)."""
    R = data.xmat[layout.body].reshape(-1, 3, 3)
    p = data.xpos[layout.body] + np.einsum("nij,nj->ni", R, layout.pos)
    n = np.einsum("nij,nj->ni", R, layout.normal)
    return p, n

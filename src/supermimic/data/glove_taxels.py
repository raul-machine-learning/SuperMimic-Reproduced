"""Glove taxels placed on the capsule hands (scripts/assets/build_capsule_hands.py).

The recording does not ship the taxel positions (config.json lists ``hand_pressure_pos.h5``, which is absent), so
each of the 129 taxels per hand is placed from the glove numbering in ``supermimic.data.sweep_mocap`` on the
palm-side surface of the capsule under it, with the outward surface normal:

  fingertip pads   on the distal capsule: rows from the tip toward the base, columns from the thumb side to the
                   little-finger side (index-finger pads 3 x 3, thumb pad 4 x 3)
  middle, proximal one taxel at the middle of that phalanx
  thumb base       on the thumb metacarpal, turned 60 deg from its palm side toward the outside of the hand
  palm rows        five rows between the finger bases and the heel of the hand, each running from the index side
                   to the little-finger side across the palm rays, projected onto the palm capsules along the palm
                   normal

The ordering inside a pad and the thumb-base side are assumptions (the glove documentation gives only the counts);
every region keeps its taxels, so region-level loads are unaffected by them. Each pad, single taxel and the palm grid
can be shifted and rolled on its capsules (``adjust_groups``); scripts/data/fit_glove_taxels.py fits those shifts
so that, on the recorded grasps, the taxels that touch the prop are the ones the glove reads as pressed
(``fitted_adjust``).

The palm side comes from anatomy alone (the thumb is on the index side; the left hand mirrors the right), not from
the hinge axes, so ``flexion_check`` can confirm that each finger flexes toward it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

FIT_FILE = Path(__file__).resolve().parents[1] / "assets/capsule_hands/glove_taxel_fit.json"

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
        self.heel = {f: self.capsule("hand", f"{f}_wrist")[0] for f in FINGERS4}

    def bid(self, short: str) -> int:
        return self.m.body(f"{self.S}_{short}").id

    def capsule(self, short: str, geom: str | None = None):
        """(proximal end, distal end, radius) of a capsule in world, proximal = the end nearer the body origin."""
        g = self.m.geom(f"{self.S}_{geom or short}_collision").id
        c, z, h, r = self.d.geom_xpos[g], self.d.geom_xmat[g].reshape(3, 3)[:, 2], self.m.geom_size[g, 1], self.m.geom_size[g, 0]
        a, b = c - h * z, c + h * z
        o = self.d.xpos[self.m.geom_bodyid[g]]
        return (a, b, r) if np.linalg.norm(a - o) <= np.linalg.norm(b - o) else (b, a, r)

    def flex_axis(self, short: str) -> np.ndarray:
        return self.d.xaxis[self.m.joint(f"{self.S}_{short}_flex").id].copy()

    def segment_palm_side(self, short: str) -> np.ndarray:
        """Direction a point on the segment moves under positive flexion of the segment's own hinge."""
        a, b, _ = self.capsule(short)
        return _unit(np.cross(self.flex_axis(short), b - a))


_POSES: dict[tuple[int, str], _Pose] = {}


def _surface(a, b, r, n, lat, axial: float, phi: float):
    """Point and normal on a capsule: ``axial`` m from the proximal end (past the distal end wraps onto its cap),
    turned ``phi`` from ``n`` toward ``lat``."""
    u = _unit(b - a)
    n = _unit(n - (n @ u) * u)
    lat = _unit(lat - (lat @ u) * u - (lat @ n) * n)
    L = np.linalg.norm(b - a)
    over = max(0.0, axial - L) / r
    d = _unit(np.cos(phi) * n + np.sin(phi) * lat + over * u)
    p = a + min(max(axial, 0.0), L) * u + r * d
    return p, d


def _capsule_gap(pts: np.ndarray, caps) -> tuple[np.ndarray, np.ndarray]:
    """Distance (...,) of points (..., 3) outside the union of the capsules, and the capsule whose surface is
    nearest each point."""
    gaps = []
    for a, b, r in caps:
        ab = b - a
        s = np.clip((pts - a) @ ab / (ab @ ab), 0.0, 1.0)
        gaps.append(np.linalg.norm(pts - (a + s[..., None] * ab), axis=-1) - r)
    gaps = np.stack(gaps, -1)
    return gaps.min(-1), np.abs(gaps).argmin(-1)


def _palm_points(pose: _Pose, base: np.ndarray, palm_caps) -> tuple[np.ndarray, np.ndarray]:
    """Exit points (N, 3) of the rays base + t * palm normal through the union of the palm capsules (bridging the
    few-mm gaps between neighbouring metacarpals, as the glove's palm does), and the capsule each lands on."""
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


def adjust_groups() -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    """Layout adjustments ``taxel_layout`` accepts, with their bounds. Pads and single taxels: (shift along the
    segment toward its tip (m), roll about the segment toward the thumb side (rad)); palm: (shift toward the finger
    bases (m), shift toward the index side (m))."""
    seg = ((-0.008, 0.008), (-1.0, 1.0))
    out = {"thumb_pad": seg, "thumb_base": seg}
    for f in FINGERS4:
        out[f"{f}_pad"], out[f"{f}_middle"], out[f"{f}_proximal"] = seg, seg, seg
    out["palm"] = ((-0.015, 0.015), (-0.012, 0.012))
    return out


def fitted_adjust() -> dict[str, tuple[float, float]]:
    """The adjustments fitted to the glove data by scripts/data/fit_glove_taxels.py (none if it has not run)."""
    return {k: tuple(v) for k, v in json.loads(FIT_FILE.read_text())["adjust"].items()} if FIT_FILE.exists() else {}


def taxel_layout(model: mujoco.MjModel, S: str, adjust: dict[str, tuple[float, float]] | None = None) -> TaxelLayout:
    """``adjust``: group -> the two values of ``adjust_groups`` (missing groups: no adjustment)."""
    pose = _POSES.get((id(model), S))
    if pose is None or pose.m is not model:
        pose = _POSES[(id(model), S)] = _Pose(model, S)
    adj = adjust or {}
    body = np.zeros(129, int)
    pos_w = np.zeros((129, 3))
    nrm_w = np.zeros((129, 3))

    def put(i: int, short: str, p: np.ndarray, n: np.ndarray) -> None:
        body[i], pos_w[i], nrm_w[i] = pose.bid(short), p, n

    def pad(first: int, short: str, rows: int, cols: int, group: str) -> None:
        a, b, r = pose.capsule(short)
        n = pose.segment_palm_side(short)
        L = np.linalg.norm(b - a)
        da, roll = adj.get(group, (0.0, 0.0))
        axial = np.linspace(L + 0.45 * r, 0.0, rows) + da
        phi = np.linspace(PAD_SPREAD, -PAD_SPREAD, cols) + roll
        for i in range(rows):
            for j in range(cols):
                put(first + i * cols + j, short, *_surface(a, b, r, n, pose.radial, axial[i], phi[j]))

    def single(i: int, short: str, group: str, phi: float = 0.0) -> None:
        a, b, r = pose.capsule(short)
        da, roll = adj.get(group, (0.0, 0.0))
        put(i, short, *_surface(a, b, r, pose.segment_palm_side(short), pose.radial, 0.5 * np.linalg.norm(b - a) + da, phi + roll))

    pad(0, "thumb3", 4, 3, "thumb_pad")
    single(12, "thumb1", "thumb_base", THUMB_BASE_TURN)
    for f, b0 in GLOVE_BASE.items():
        pad(b0, f"{f}3", 3, 3, f"{f}_pad")
        single(b0 + 9, f"{f}2", f"{f}_middle")
        single(b0 + 10, f"{f}1", f"{f}_proximal")
    p_along, p_lat = adj.get("palm", (0.0, 0.0))
    along = pose.mcp["middle"] - pose.heel["middle"]
    along = along - (along @ pose.palm_n) * pose.palm_n
    palm_shift = p_along * along / np.linalg.norm(along) + p_lat * pose.radial

    palm_shorts = ([(f"{f}0", None) for f in FINGERS4] + [("hand", f"{f}_wrist") for f in FINGERS4]
                   + [("hand", "heel"), ("hand", "thenar")])
    palm_caps = [pose.capsule(s, g) for s, g in palm_shorts]
    ids, bases = [], []
    for (lo, hi), frac in zip(PALM_ROWS, PALM_ROW_FRACTION):
        line = [pose.heel[f] + frac * (pose.mcp[f] - pose.heel[f]) for f in FINGERS4]
        for i, s in enumerate(np.linspace(0.0, 3.0, hi - lo)):
            k = min(int(s), 2)
            w = s - k
            ids.append(lo + i)
            bases.append((1 - w) * line[k] + w * line[k + 1] + palm_shift - 0.03 * pose.palm_n)
    pts, which = _palm_points(pose, np.array(bases), palm_caps)
    for i, p, cap in zip(ids, pts, which):
        put(i, palm_shorts[int(cap)][0], p, pose.palm_n)

    pos = np.zeros_like(pos_w)
    nrm = np.zeros_like(nrm_w)
    for i in range(129):
        R = pose.d.xmat[body[i]].reshape(3, 3)
        pos[i] = R.T @ (pos_w[i] - pose.d.xpos[body[i]])
        nrm[i] = R.T @ nrm_w[i]
    return TaxelLayout(body=body, pos=pos, normal=nrm)


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

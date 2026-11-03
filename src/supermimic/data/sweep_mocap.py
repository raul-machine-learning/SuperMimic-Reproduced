"""Readers for the HiPHI-OM sweep recording in ``data/sweep_mocap`` (human_bones / trackers_sixdof / hand_pressure).

Everything returned is in MuJoCo conventions at the recording's own scale: Z-up world, meters, quaternions
(w, x, y, z). The recording is Y-up with the actor's rest pose facing +Z; the conversion is a +90 deg rotation about
X, (x, y, z) -> (x, -z, y), which keeps the frame right-handed.

Glove pressure: 129 taxels per hand, identical numbering on both gloves (palm view, the right glove mirrored):
  0-11      thumb tip pad (4 x 3 grid)
  12        thumb base, on the side of the hand
  13-21 / 22 / 23   index: tip pad (3 x 3), middle phalanx, proximal phalanx
  24-32 / 33 / 34   middle finger
  35-43 / 44 / 45   ring finger
  46-54 / 55 / 56   little finger
  57-68, 69-83, 84-98, 99-113, 114-128   palm rows, from the finger bases down to the heel of the hand; each row runs
                    from the index side to the little-finger side (split in halves here; an odd middle taxel goes to the
                    little-finger half)
Readings are raw (0..~53, not newtons) and never fall to zero: the glove's fit leaves a resting load. Contact is the
excess over a per-taxel baseline taken while the hands hold nothing (the first 2 s, before the broom and dustpan
leave the table). Taxels 84-98 never exceed 3 on either glove in this clip.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

Y_UP_TO_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
_C = Rotation.from_matrix(Y_UP_TO_Z_UP)

PROP_MESHES = {
    "cleaningtools_broom": "cleaningtools_broom.stl",
    "cleaningtools_dustpan": "cleaningtools_dustpan.stl",
    "TrashBucket_chip": "trashbucket_chip.stl",
}
# Recorded table top, where all three props rest (task_info says 90 cm). Its footprint is not in the recording.
TABLE_TOP_HEIGHT = 0.883


@dataclass
class HumanMotion:
    times: np.ndarray  # (T,) seconds from the first frame
    names: list[str]  # bone names, including the *_End leaf sites
    parents: list[int]  # index into names, -1 for Hips
    pos: np.ndarray  # (T, B, 3) world positions
    quat: np.ndarray  # (T, B, 4) world orientations (w, x, y, z)
    offsets: dict[str, np.ndarray]  # rest-pose offset of each bone in its parent's frame

    def index(self, name: str) -> int:
        return self.names.index(name)

    def p(self, name: str) -> np.ndarray:
        return self.pos[:, self.index(name)]


@dataclass
class PropMotion:
    times: np.ndarray  # (T,)
    pos: dict[str, np.ndarray]  # name -> (T, 3)
    quat: dict[str, np.ndarray]  # name -> (T, 4) (w, x, y, z)


@dataclass
class HandPressure:
    times: np.ndarray  # (T,) seconds, same clock origin as HumanMotion.times
    left: np.ndarray  # (T, 129) raw uint8 taxel readings
    right: np.ndarray  # (T, 129)


def _decode(n) -> str:
    return n.decode() if isinstance(n, bytes) else str(n)


def _to_z_up_rot(q_wxyz: np.ndarray) -> np.ndarray:
    r = _C * Rotation.from_quat(q_wxyz.reshape(-1, 4), scalar_first=True)
    return r.as_quat(scalar_first=True).reshape(q_wxyz.shape)


def load_human(mocap_dir: Path) -> HumanMotion:
    """Forward kinematics of human_bones.h5.

    Per frame, Hips carries its world pose and every other bone its pose relative to its parent (BVH semantics).
    Frame positions are in cm even though the Skeleton group's offsets (used only for the *_End leaves) are in m.
    """
    with h5py.File(mocap_dir / "human_bones.h5", "r") as f:
        sk = f["Skeleton"]
        offset_unit = {"m": 1.0, "cm": 0.01}[_decode(sk.attrs.get("unit", "m"))]
        meta = {
            n: (_decode(sk[n].attrs["parent"]), np.asarray(sk[n].attrs["offset"], float) * offset_unit, bool(sk[n].attrs.get("is_end", False)))
            for n in sk.keys()
        }
        data = f["data"][:]

    order: list[str] = []
    pending = ["Hips"]
    children: dict[str, list[str]] = {}
    for n, (par, _, _) in meta.items():
        children.setdefault(par, []).append(n)
    while pending:
        n = pending.pop(0)
        order.append(n)
        pending.extend(sorted(children.get(n, [])))
    parents = [order.index(meta[n][0]) if meta[n][0] else -1 for n in order]

    T, B = len(data), len(order)
    local_p = np.zeros((T, B, 3))
    local_q = np.tile([1.0, 0.0, 0.0, 0.0], (T, B, 1))
    for b, n in enumerate(order):
        if meta[n][2]:
            local_p[:, b] = meta[n][1]
    col = {n: b for b, n in enumerate(order)}
    for t, row in enumerate(data):
        for e in row["elements"]:
            b = col[_decode(e["name"])]
            local_p[t, b] = np.asarray(e["position"], float) * 0.01
            local_q[t, b] = e["rotation"]

    world_p = np.zeros_like(local_p)
    world_r: list[Rotation] = [None] * B  # type: ignore[list-item]
    for b in range(B):
        r_local = Rotation.from_quat(local_q[:, b], scalar_first=True)
        if parents[b] < 0:
            world_p[:, b] = local_p[:, b]
            world_r[b] = r_local
        else:
            rp = world_r[parents[b]]
            world_p[:, b] = world_p[:, parents[b]] + rp.apply(local_p[:, b])
            world_r[b] = rp * r_local

    pos = world_p @ Y_UP_TO_Z_UP.T
    quat = np.stack([(_C * r).as_quat(scalar_first=True) for r in world_r], axis=1)
    times = data["timestamp"] - data["timestamp"][0]
    offsets = {n: meta[n][1] for n in order}
    return HumanMotion(times=times, names=order, parents=parents, pos=pos, quat=quat, offsets=offsets)


def load_props(mocap_dir: Path, names=tuple(PROP_MESHES)) -> PropMotion:
    """Prop poses from trackers_sixdof.h5 (meters, w-first quaternions; the STL meshes share each tracker's frame)."""
    with h5py.File(mocap_dir / "trackers_sixdof.h5", "r") as f:
        data = f["data"][:]
    T = len(data)
    pos = {n: np.full((T, 3), np.nan) for n in names}
    quat = {n: np.full((T, 4), np.nan) for n in names}
    for t, row in enumerate(data):
        for e in row["elements"]:
            n = _decode(e["name"])
            if n in pos:
                pos[n][t] = e["position"]
                quat[n][t] = e["rotation"]
    for n in names:
        pos[n] = _fill_gaps(pos[n]) @ Y_UP_TO_Z_UP.T
        quat[n] = _to_z_up_rot(_fill_gaps(quat[n]))
    return PropMotion(times=data["timestamp"] - data["timestamp"][0], pos=pos, quat=quat)


def load_pressure(mocap_dir: Path) -> HandPressure | None:
    path = mocap_dir / "hand_pressure.h5"
    if not path.exists():
        return None
    with h5py.File(mocap_dir / "human_bones.h5", "r") as f:
        t0 = float(f["data"][0]["timestamp"])
    with h5py.File(path, "r") as f:
        data = f["data"][:]
    left = np.zeros((len(data), 129), np.uint8)
    right = np.zeros((len(data), 129), np.uint8)
    for t, row in enumerate(data):
        for e in row["elements"]:
            side = _decode(e["name"]).lower()
            (left if side.startswith("l") else right)[t] = e["value"]
    return HandPressure(times=data["timestamp"] - t0, left=left, right=right)


def _fill_gaps(x: np.ndarray) -> np.ndarray:
    """Hold the last valid sample over dropped tracker frames (and back-fill a leading gap)."""
    x = x.copy()
    bad = np.isnan(x).any(axis=1)
    if bad.all():
        raise ValueError("tracker never observed")
    idx = np.where(~bad, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    first = np.argmax(~bad)
    idx[:first] = first
    return x[idx]


def estimate_table(prop_pos: dict[str, np.ndarray], hips_pos: np.ndarray, margin: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    """Box (center, half-size) under every tabletop position of the props, kept clear of the actor's hips."""
    pts = np.concatenate([p for p in prop_pos.values()])
    pts = pts[np.abs(pts[:, 2] - TABLE_TOP_HEIGHT) < 0.20]
    lo, hi = pts[:, :2].min(0) - margin, pts[:, :2].max(0) + margin
    hips_xy = hips_pos[:, :2]
    to_table = (lo + hi) / 2 - hips_xy.mean(0)
    axis = int(np.argmax(np.abs(to_table)))
    if to_table[axis] < 0:
        hi[axis] = min(hi[axis], hips_xy[:, axis].min() - 0.20)
    else:
        lo[axis] = max(lo[axis], hips_xy[:, axis].max() + 0.20)
    half = np.array([*(hi - lo) / 2, TABLE_TOP_HEIGHT / 2])
    center = np.array([*(hi + lo) / 2, TABLE_TOP_HEIGHT / 2])
    return center, half


REGIONS: dict[str, tuple[int, ...]] = {"thumb_tip": tuple(range(0, 12)), "thumb_base": (12,)}
for _finger, _b in (("index", 13), ("middle", 24), ("ring", 35), ("little", 46)):
    REGIONS[f"{_finger}_tip"] = tuple(range(_b, _b + 9)) + (_b + 9,)  # tip pad + middle phalanx
    REGIONS[f"{_finger}_proximal"] = (_b + 10,)
_ROWS = ((57, 69), (69, 84), (84, 99), (99, 114), (114, 129))


def _palm(rows: tuple[int, ...], side: str) -> tuple[int, ...]:
    out: list[int] = []
    for r in rows:
        a, b = _ROWS[r]
        half = (b - a) // 2
        out += range(a, a + half) if side == "index" else range(a + half, b)
    return tuple(out)


REGIONS["palm_upper_index_side"] = _palm((0, 1), "index")
REGIONS["palm_upper_little_side"] = _palm((0, 1), "little")
REGIONS["palm_middle"] = tuple(range(84, 99))
REGIONS["palm_lower_index_side"] = _palm((3, 4), "index")
REGIONS["palm_lower_little_side"] = _palm((3, 4), "little")
assert sorted(i for v in REGIONS.values() for i in v) == list(range(129))


def region_loaded(raw: np.ndarray, p_times: np.ndarray, times: np.ndarray, rest: tuple[float, float] = (0.0, 2.0),
                  rel: float = 0.15, floor: float = 3.0) -> np.ndarray:
    """(T, len(REGIONS)) bool on ``times``: a region's load above the rest baseline exceeds ``rel`` of its clip
    maximum and ``floor`` raw units."""
    sel = (p_times >= rest[0]) & (p_times < rest[1])
    if not sel.any():
        raise ValueError(f"no pressure samples in the rest window {rest}")
    load = np.clip(raw.astype(np.float64) - raw[sel].astype(np.float64).mean(0), 0.0, None)
    load = np.stack([load[:, list(idx)].sum(1) for idx in REGIONS.values()], 1)
    load = np.stack([np.interp(times, p_times, load[:, r]) for r in range(load.shape[1])], 1)
    return load > np.maximum(rel * load.max(0), floor)

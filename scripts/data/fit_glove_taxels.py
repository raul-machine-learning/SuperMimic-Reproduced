"""Fit the glove taxel layout (supermimic.data.glove_taxels) so that, on the recorded grasps, the taxels touching the
held prop are the ones the glove reads as pressed.

  uv run python scripts/data/fit_glove_taxels.py

Runs on the grasp-corrected clip (hands registered to their props by register_hands_to_props.py, fingers closed onto
them by close_capsule_fingers.py), replayed kinematically. A taxel touches when its signed distance to the prop
surface (the convex collision pieces, sampled densely) is under ``touch``; the glove presses a taxel when its load
above the resting baseline is a sizable share of its clip maximum (``supermimic.data.sweep_mocap.taxel_pressed``).

The fit moves each fingertip pad, single taxel and the palm grid on its capsules within the bounds of
``adjust_groups``, one set of adjustments for both hands (the gloves share their layout, mirrored), minimizing a
class-balanced cross-entropy between a soft touch (sigmoid of the signed distance) and the glove's pressed flags, plus
a small pull toward the unadjusted layout. Reports per-region agreement before and after and writes the
adjustments to ``supermimic/assets/capsule_hands/glove_taxel_fit.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import tyro
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from supermimic.data.glove_taxels import FIT_FILE, adjust_groups, taxel_layout
from supermimic.data.sweep_mocap import REGIONS, taxel_pressed
from supermimic.robots import HAND_DATA, HANDS_XML

ROOT = Path(__file__).resolve().parents[2]
SIDES = ("L", "R")
SIDE_NAME = {"L": "left", "R": "right"}
HELD = {"L": "cleaningtools_broom", "R": "cleaningtools_dustpan"}


def prop_cloud(props_dir: Path, name: str, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Surface points and outward normals of a prop's convex pieces, in the prop's frame."""
    spec = mujoco.MjSpec()
    b = spec.worldbody.add_body(name=name)
    files = sorted((props_dir / name).glob("collision_*.obj"))
    for i, f in enumerate(files):
        spec.add_mesh(name=f"m{i}", file=str(f.resolve()))
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"m{i}")
    m = spec.compile()
    rng = np.random.default_rng(0)
    pts, nrm = [], []
    for g in range(m.ngeom):
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        f = m.mesh_face[m.mesh_faceadr[mid] : m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
        R = Rotation.from_quat(m.geom_quat[g], scalar_first=True).as_matrix()
        v = v @ R.T + m.geom_pos[g]
        a, b_, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        n = np.cross(b_ - a, c - a)
        area = 0.5 * np.linalg.norm(n, axis=1)
        n = n / np.maximum(2 * area, 1e-12)[:, None]
        n *= np.sign(np.einsum("ij,ij->i", (a + b_ + c) / 3 - v.mean(0), n))[:, None]  # convex: outward = away from the centroid
        k = np.maximum(1, np.round(area / spacing**2)).astype(int)
        tri = np.repeat(np.arange(len(f)), k)
        u, w = rng.random(len(tri)), rng.random(len(tri))
        flip = u + w > 1
        u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
        pts.append(a[tri] + u[:, None] * (b_[tri] - a[tri]) + w[:, None] * (c[tri] - a[tri]))
        nrm.append(n[tri])
    return np.concatenate(pts), np.concatenate(nrm)


def main(clip: Path = HAND_DATA / "sweep_task023_hands_registered_closed.npz", props_dir: Path = ROOT / "data/props",
         touch: float = 0.005, soft: float = 0.002, max_frames: int = 200, prior: float = 0.05, spacing: float = 0.0015,
         write: bool = True) -> None:
    c = dict(np.load(clip))
    model = mujoco.MjModel.from_xml_path(str(HANDS_XML))
    data = mujoco.MjData(model)
    names = [str(n) for n in c["joint_names"]]
    qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in names])
    root = {S: model.jnt_qposadr[model.joint(f"{S}_hand_root").id] for S in SIDES}
    prop_names = [str(n) for n in c["prop_names"]]
    T = len(c["times"])
    pressed = {S: taxel_pressed(c[f"pressure_{SIDE_NAME[S]}"], c["pressure_times"], c["times"]) for S in SIDES}
    clouds = {S: prop_cloud(props_dir, HELD[S], spacing) for S in SIDES}
    trees = {S: cKDTree(clouds[S][0]) for S in SIDES}
    base = {S: taxel_layout(model, S) for S in SIDES}

    # Hand body poses in the held prop's frame, on the grasp frames (some taxel pressed, the hand within 3 cm).
    frames, body_pose = {}, {}
    for k, S in enumerate(SIDES):
        cand = np.nonzero(pressed[S].any(1))[0]
        cand = cand[np.linspace(0, len(cand) - 1, min(max_frames * 2, len(cand))).astype(int)]
        j = prop_names.index(HELD[S])
        xs, Rs, keep = [], [], []
        for t in cand:
            data.qpos[qadr] = c["joint_pos"][t]
            for kk, s in enumerate(SIDES):
                a = root[s]
                data.qpos[a : a + 3], data.qpos[a + 3 : a + 7] = c["base_pos"][t, kk], c["base_quat"][t, kk]
            mujoco.mj_kinematics(model, data)
            Rp = Rotation.from_quat(c["prop_quat"][t, j], scalar_first=True).as_matrix()
            x = (data.xpos - c["prop_pos"][t, j]) @ Rp
            R = np.einsum("ji,bjk->bik", Rp, data.xmat.reshape(-1, 3, 3))
            p = x[base[S].body] + np.einsum("nij,nj->ni", R[base[S].body], base[S].pos)
            if trees[S].query(p)[0].min() < 0.03:
                keep.append(t)
                xs.append(x)
                Rs.append(R)
        keep = np.array(keep)
        sel = np.linspace(0, len(keep) - 1, min(max_frames, len(keep))).astype(int)
        frames[S] = keep[sel]
        body_pose[S] = (np.stack(xs)[sel], np.stack(Rs)[sel])
        print(f"{SIDE_NAME[S]} hand: {len(frames[S])} grasp frames of {T}")

    def signed_dist(S: str, layout) -> np.ndarray:
        """(F, 129) signed distance of the taxels to the prop surface (negative inside)."""
        x, R = body_pose[S]
        p = x[:, layout.body] + np.einsum("fnij,nj->fni", R[:, layout.body], layout.pos)
        d, i = trees[S].query(p.reshape(-1, 3))
        q, n = clouds[S][0][i], clouds[S][1][i]
        s = np.einsum("ij,ij->i", p.reshape(-1, 3) - q, n)
        return np.where(s < 0, -d, d).reshape(p.shape[:2])

    groups = adjust_groups()
    keys = list(groups)
    lo = np.array([b[0] for k in keys for b in groups[k]])
    hi = np.array([b[1] for k in keys for b in groups[k]])

    def unpack(z: np.ndarray) -> dict:
        v = lo + (hi - lo) * (np.tanh(z) + 1) / 2
        return {k: (float(v[2 * i]), float(v[2 * i + 1])) for i, k in enumerate(keys)}

    z0 = np.arctanh(np.clip(-(lo + hi) / (hi - lo), -0.999, 0.999))  # zero adjustment

    def loss(z: np.ndarray) -> float:
        adj = unpack(z)
        tot = 0.0
        for S in SIDES:
            s = signed_dist(S, taxel_layout(model, S, adj, roll=True))
            y = pressed[S][frames[S]]
            pr = np.clip(1 / (1 + np.exp((s - touch) / soft)), 1e-6, 1 - 1e-6)
            w_pos = 0.5 / max(y.mean(), 1e-6)
            w_neg = 0.5 / max(1 - y.mean(), 1e-6)
            tot += float(np.mean(-(w_pos * y * np.log(pr) + w_neg * (1 - y) * np.log(1 - pr))))
        v = lo + (hi - lo) * (np.tanh(z) + 1) / 2
        return tot + prior * float(np.sum((v / np.maximum(np.abs(lo), np.abs(hi))) ** 2))

    def report(tag: str, adj: dict) -> dict:
        print(tag)
        out = {}
        for S in SIDES:
            s = signed_dist(S, taxel_layout(model, S, adj, roll=True))
            y = pressed[S][frames[S]]
            hit = s < touch
            print(f"  {SIDE_NAME[S]} hand ({HELD[S].removeprefix('cleaningtools_')}): pressed taxels touching (recall), touching taxels pressed (precision)")
            rows = {}
            for reg, idx in REGIONS.items():
                yy, hh = y[:, list(idx)], hit[:, list(idx)]
                if yy.sum() == 0 and hh.sum() == 0:
                    continue
                rec = (yy & hh).sum() / max(yy.sum(), 1)
                pre = (yy & hh).sum() / max(hh.sum(), 1)
                rows[reg] = (float(rec), float(pre), int(yy.sum()), int(hh.sum()))
                print(f"    {reg:24s} recall {rec:4.0%} of {yy.sum():5d}   precision {pre:4.0%} of {hh.sum():5d}")
            rec = (y & hit).sum() / max(y.sum(), 1)
            pre = (y & hit).sum() / max(hit.sum(), 1)
            f1 = 2 * rec * pre / max(rec + pre, 1e-9)
            print(f"    {'all taxels':24s} recall {rec:4.0%} of {y.sum():5d}   precision {pre:4.0%} of {hit.sum():5d}   F1 {f1:.2f}")
            out[S] = dict(regions=rows, recall=float(rec), precision=float(pre), f1=float(f1))
        return out

    before = report("before the fit (layout from the glove numbering)", {})
    res = minimize(loss, z0, method="Powell", options=dict(maxiter=20, xtol=1e-2, ftol=1e-4))
    adj = unpack(res.x)
    after = report(f"after the fit (loss {loss(z0):.3f} -> {res.fun:.3f})", adj)
    print("adjustments (shift mm, roll deg | palm: toward finger bases mm, toward index side mm)")
    for k, (a, b) in adj.items():
        print(f"  {k:18s} {a * 1000:+5.1f} mm  " + (f"{b * 1000:+5.1f} mm" if k == "palm" else f"{np.degrees(b):+5.0f} deg"))
    if write:
        FIT_FILE.write_text(json.dumps(dict(adjust=adj, clip=str(clip), touch=touch, before=before, after=after), indent=1))
        print(f"wrote {FIT_FILE}")


if __name__ == "__main__":
    tyro.cli(main)

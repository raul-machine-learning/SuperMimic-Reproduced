"""How close the retargeted Inspire hands come to the props while the glove reports contact.

  SM_ROBOT=g1_inspire uv run python scripts/data/audit_hand_prop_contact.py --retarget ~/g1-sweep-retarget/out/sweep_g1_inspire.npz \
      --mocap-dir data/sweep_mocap

Replays the retarget (robot qpos, prop poses, glove pressure) kinematically on SuperMimic's g1_inspire model and,
per frame, measures the gap between each glove region's Inspire link (its collision primitives, i.e. what the
physics will see) and the nearest prop surface. A grasp can only come out of tracking the reference if the loaded
regions are within a few millimetres of a prop. Distances are unsigned: penetration reads as a small gap too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import tyro
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
RETARGET_ROOT = Path.home() / "g1-sweep-retarget"


def _load_regions(retarget_root: Path):
    sys.path.insert(0, str(retarget_root))
    from sweep_retarget.pressure import INSPIRE_LINK, REGIONS, contact_load, region_load

    return REGIONS, INSPIRE_LINK, contact_load, region_load


def geom_surface_points(model: mujoco.MjModel, g: int, n: int = 12) -> np.ndarray:
    """Points on the surface of a box / capsule / sphere geom, in the geom frame."""
    t, s = int(model.geom_type[g]), model.geom_size[g]
    if t == int(mujoco.mjtGeom.mjGEOM_BOX):
        u = np.linspace(-1, 1, n)
        a, b = np.meshgrid(u, u)
        a, b = a.ravel(), b.ravel()
        pts = []
        for ax in range(3):
            o1, o2 = [i for i in range(3) if i != ax]
            for sign in (-1, 1):
                p = np.zeros((a.size, 3))
                p[:, ax], p[:, o1], p[:, o2] = sign, a, b
                pts.append(p)
        return np.concatenate(pts) * s[:3]
    if t in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_SPHERE)):
        r, h = s[0], (s[1] if t == int(mujoco.mjtGeom.mjGEOM_CAPSULE) else 0.0)
        d = np.random.default_rng(0).normal(size=(n * n * 4, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        p = r * d
        p[:, 2] += np.sign(d[:, 2]) * h
        return p
    raise ValueError(f"geom type {t} not supported")


def main(retarget: Path, mocap_dir: Path = ROOT / "data/sweep_mocap", retarget_root: Path = RETARGET_ROOT,
         rest: tuple[float, float] = (0.0, 2.0), loaded_frac: float = 0.15, prop_shift_z: float = 0.0, out: Path | None = None) -> None:
    """``out``: also save the per-frame gaps and glove-load flags (npz, for scripts/media/render_contact_audit.py)."""
    from supermimic.robots import ROBOT

    assert ROBOT.name == "g1_inspire", "run with SM_ROBOT=g1_inspire"
    REGIONS, INSPIRE_LINK, contact_load, region_load = _load_regions(retarget_root)
    z = np.load(retarget)
    model = ROBOT.mj_model()
    data = mujoco.MjData(model)
    names = [str(n) for n in z["joint_names"]]
    assert tuple(names) == ROBOT.joint_names
    times, qpos, scale = z["times"], z["qpos"], float(z["scale"])
    prop_names = [str(n) for n in z["prop_names"]][:2]  # broom, dustpan (the trash can is never touched)
    prop_pos, prop_quat = z["prop_pos"][:, :2].copy(), z["prop_quat"][:, :2]
    prop_pos[..., 2] += prop_shift_z

    trees = []
    for n in prop_names:
        mesh = trimesh.load(mocap_dir / f"{n}.stl")
        mesh.apply_scale(0.01 * scale)
        pts, _ = trimesh.sample.sample_surface_even(mesh, 60000, seed=0)
        trees.append(cKDTree(pts))

    # Region -> collision geoms of its Inspire link, per hand.
    region_geoms: dict[tuple[str, str], list[int]] = {}
    for side, s in (("left", "L"), ("right", "R")):
        for reg in REGIONS:
            bid = model.body(f"{s}_{INSPIRE_LINK[reg]}").id
            region_geoms[(side, reg)] = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid and model.geom_contype[g]]
    local = {g: geom_surface_points(model, g) for gs in region_geoms.values() for g in gs}

    T = len(times)
    dist = {k: np.zeros((T, len(prop_names))) for k in region_geoms}  # (T, prop)
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        for k, gs in region_geoms.items():
            pw = np.concatenate([data.geom_xpos[g] + local[g] @ data.geom_xmat[g].reshape(3, 3).T for g in gs])
            for j in range(len(prop_names)):
                R = Rotation.from_quat(prop_quat[t, j], scalar_first=True)
                pl = R.inv().apply(pw - prop_pos[t, j])
                dist[k][t, j] = trees[j].query(pl)[0].min()

    pt = z["pressure_times"]
    saved: dict[str, np.ndarray] = {}
    print(f"{retarget.name}: {T} frames, scene scale {scale:.3f}, props shifted by {prop_shift_z * 100:+.1f} cm in z")
    for side in ("left", "right"):
        load = region_load(contact_load(z[f"pressure_{side}"], pt, rest))
        load = np.stack([np.interp(times, pt, load[:, r]) for r in range(load.shape[1])], 1)
        total = load.sum(1)
        held = total > loaded_frac * total.max()
        hand_min = np.min([dist[(side, r)] for r in REGIONS], axis=0)  # (T, prop)
        j = int(np.argmin(np.median(hand_min[held], 0)))
        saved[f"{side}_prop"] = np.array(j)
        saved[f"{side}_gap"] = np.stack([dist[(side, r)][:, j] for r in REGIONS], 1)
        saved[f"{side}_loaded"] = np.stack([load[:, r] > max(0.15 * load[:, r].max(), 3.0) for r in range(len(REGIONS))], 1)
        print(f"\n{side} hand: glove loaded on {held.mean():.0%} of frames; holds the {prop_names[j]}")
        print(f"  whole hand to {prop_names[j]}: loaded median {np.median(hand_min[held, j]) * 100:.1f} cm, "
              f"p90 {np.percentile(hand_min[held, j], 90) * 100:.1f} cm; unloaded median {np.median(hand_min[~held, j]) * 100:.1f} cm")
        print(f"  {'region':24s} {'loaded on':>9s}  {'gap when loaded: median':>24s}  {'p90':>6s}")
        for r, reg in enumerate(REGIONS):
            on = load[:, r] > max(0.15 * load[:, r].max(), 3.0)
            if on.sum() < 10:
                continue
            d = dist[(side, reg)][on, j] * 100
            print(f"  {reg:24s} {on.mean():9.0%}  {np.median(d):21.1f} cm  {np.percentile(d, 90):4.1f} cm")
    if out is not None:
        links = np.array([INSPIRE_LINK[r] for r in REGIONS])
        np.savez(out, times=times, regions=np.array(list(REGIONS)), links=links, prop_names=np.array(prop_names), prop_shift_z=prop_shift_z, **saved)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    tyro.cli(main)

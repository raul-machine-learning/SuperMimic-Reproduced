"""Do the glove taxels (supermimic.data.glove_taxels) face the prop the hand holds, on the frames the glove is loaded?

  uv run python scripts/data/check_taxel_direction.py

Kinematic replay of the mocap clip. On every frame where a glove region is loaded, for each of its taxels: the
distance from the taxel to the held prop (convex pieces from scripts/assets/build_props.py) and the cosine between
the taxel's outward normal and the direction to the nearest prop point. Per region, the taxel nearest the prop
decides the frame: "front" if the prop is within 45 deg of its normal (or the taxel is inside the prop), "behind" if
it is more than 135 deg away.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import tyro

from supermimic.data.glove_taxels import glove_layout
from supermimic.data.sweep_mocap import REGIONS, region_loaded
from supermimic.robots import HAND_DATA, HANDS_XML

ROOT = Path(__file__).resolve().parents[2]
SIDES = ("L", "R")
SIDE_NAME = {"L": "left", "R": "right"}
HELD = {"L": "cleaningtools_broom", "R": "cleaningtools_dustpan"}
TAXEL_R = 0.001


def build(props_dir: Path) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(HANDS_XML))
    base = spec.compile()
    for S in SIDES:
        lay = glove_layout(base, S)
        for i in range(129):
            spec.body(base.body(int(lay.body[i])).name).add_geom(
                name=f"{S}_taxel{i:03d}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[TAXEL_R, 0, 0], pos=lay.pos[i],
                contype=0, conaffinity=0, group=4)
    for n in set(HELD.values()):
        b = spec.worldbody.add_body(name=n, mocap=True)
        for i, f in enumerate(sorted((props_dir / n).glob("collision_*.obj"))):
            spec.add_mesh(name=f"{n}_c{i:02d}", file=str(f.resolve()))
            b.add_geom(name=f"{n}_c{i:02d}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{n}_c{i:02d}", contype=0, conaffinity=0)
    return spec.compile()


def main(clip: Path = HAND_DATA / "sweep_task023_hands.npz", props_dir: Path = ROOT / "data/props", reach: float = 0.05) -> None:
    c = dict(np.load(clip))
    model = build(props_dir)
    data = mujoco.MjData(model)
    lay = {S: glove_layout(model, S) for S in SIDES}
    names = [str(n) for n in c["joint_names"]]
    qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in names])
    root = {S: model.jnt_qposadr[model.joint(f"{S}_hand_root").id] for S in SIDES}
    prop_names = [str(n) for n in c["prop_names"]]
    taxel_g = {S: np.array([model.geom(f"{S}_taxel{i:03d}").id for i in range(129)]) for S in SIDES}
    pieces = {S: [g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body(HELD[S]).id] for S in SIDES}
    loaded = {S: region_loaded(c[f"pressure_{SIDE_NAME[S]}"], c["pressure_times"], c["times"]) for S in SIDES}
    regions = list(REGIONS)
    T = len(c["times"])
    gap = {S: np.full((T, 129), np.nan) for S in SIDES}
    cos = {S: np.full((T, 129), np.nan) for S in SIDES}
    fromto = np.zeros(6)
    for t in range(T):
        if not (loaded["L"][t].any() or loaded["R"][t].any()):
            continue
        data.qpos[qadr] = c["joint_pos"][t]
        for k, S in enumerate(SIDES):
            a = root[S]
            data.qpos[a : a + 3], data.qpos[a + 3 : a + 7] = c["base_pos"][t, k], c["base_quat"][t, k]
        for S in SIDES:
            m = model.body_mocapid[model.body(HELD[S]).id]
            j = prop_names.index(HELD[S])
            data.mocap_pos[m], data.mocap_quat[m] = c["prop_pos"][t, j], c["prop_quat"][t, j]
        mujoco.mj_kinematics(model, data)
        for S in SIDES:
            idx = sorted({i for r, reg in enumerate(regions) if loaded[S][t, r] for i in REGIONS[reg]})
            for i in idx:
                g1 = taxel_g[S][i]
                R = data.xmat[model.geom_bodyid[g1]].reshape(3, 3)
                n = R @ lay[S].normal[i]
                best, v = reach, None
                for g2 in pieces[S]:
                    if np.linalg.norm(data.geom_xpos[g1] - data.geom_xpos[g2]) - model.geom_rbound[g2] > best:
                        continue
                    d = mujoco.mj_geomDistance(model, data, g1, g2, best, fromto)
                    if d < best:
                        best, v = d, fromto[3:] - fromto[:3]
                gap[S][t, i] = best + TAXEL_R
                if v is not None:
                    cos[S][t, i] = 1.0 if best < 0 else float(n @ v / max(np.linalg.norm(v), 1e-9))

    print("taxel direction check (kinematic replay): per loaded glove region, the taxel nearest the held prop")
    for S in SIDES:
        print(f"  {SIDE_NAME[S]} hand, {HELD[S].removeprefix('cleaningtools_')}")
        print(f"    {'region':24s} {'loaded':>6s}  {'gap median':>10s}  {'front':>5s} {'side':>5s} {'behind':>6s} {'> 5 cm':>6s}")
        tot = np.zeros(4)
        for r, reg in enumerate(regions):
            on = loaded[S][:, r]
            if not on.any():
                continue
            idx = list(REGIONS[reg])
            g = gap[S][on][:, idx]
            cc = cos[S][on][:, idx]
            near = np.nanargmin(np.where(np.isnan(g), np.inf, g), 1)
            gn = g[np.arange(len(g)), near]
            cn = cc[np.arange(len(g)), near]
            far = np.isnan(cn)
            front, behind = (cn > np.cos(np.radians(45))) & ~far, (cn < np.cos(np.radians(135))) & ~far
            side = ~far & ~front & ~behind
            tot += [front.sum(), side.sum(), behind.sum(), far.sum()]
            print(f"    {reg:24s} {on.mean():6.0%}  {np.median(gn) * 1000:7.1f} mm  {front.mean():5.0%} {side.mean():5.0%} {behind.mean():6.0%} {far.mean():6.0%}")
        f = tot / tot.sum()
        print(f"    {'all loaded region-frames':24s} {'':6s}  {'':10s}  {f[0]:5.0%} {f[1]:5.0%} {f[2]:6.0%} {f[3]:6.0%}")


if __name__ == "__main__":
    tyro.cli(main)

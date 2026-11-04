"""Before/after picture of a shifted glove taxel layout on the capsule hands at a mocap frame.

  SM_ROBOT=capsule_hands_props MUJOCO_GL=egl uv run python scripts/media/render_taxel_shift.py --out /tmp/taxel_shift.png

For each hand, two panels seen from the palm side with the held prop see-through: the fitted layout and the layout
with ``shift`` (group -> mm added along the hand, as scripts/diag_taxel_shift.py: palm toward the finger bases,
finger groups toward the tip). Taxels are colored by the glove's pressure at that frame (grey: none, yellow to red:
share of the hand's peak); taxels within 5 mm of the prop are ringed in black.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import tyro
from PIL import Image, ImageDraw, ImageFont

from supermimic.data.glove_taxels import fitted_adjust, glove_layout, taxel_layout, world_taxels
from supermimic.data.sweep_mocap import taxel_load
from supermimic.robots import HAND_DATA

ROOT = Path(__file__).resolve().parents[2]
HELD = {"L": "cleaningtools_broom", "R": "cleaningtools_dustpan"}
FINGER_GROUPS = ("thumb_pad", "thumb_base", *(f"{f}_{k}" for f in ("index", "middle", "ring", "pinky") for k in ("pad", "middle", "proximal")))


def _rch():
    spec = importlib.util.spec_from_file_location("rch", ROOT / "scripts/media/render_capsule_hands.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def shifted(shift_mm: dict[str, float]) -> dict:
    out = dict(fitted_adjust())
    for k, mm in shift_mm.items():
        for g in (FINGER_GROUPS if k == "fingers" else (k,)):
            a, r = out.get(g, (0.0, 0.0))
            out[g] = (a + mm / 1000.0, r)
    return out


def main(out: Path, clip: Path = HAND_DATA / "sweep_task023_hands_registered_closed.npz", time: float = 7.0,
         left_shift: str = "palm=-10", right_shift: str = "fingers=-8", left_scores: str = "0.235 -> 0.256",
         right_scores: str = "0.172 -> 0.191", tile: int = 640) -> None:
    """``*_shift``: comma-separated group=mm (``palm``, ``fingers`` or a layout group); ``*_scores``: caption text."""
    rch = _rch()
    c = dict(np.load(clip))
    model, props = rch.build_model(c, Path(str(c["mocap_dir"])), ghosts=False)
    data = mujoco.MjData(model)
    t = int(np.argmin(np.abs(c["times"] - (c["times"][0] + time))))
    for k, S in enumerate("LR"):
        a = model.jnt_qposadr[model.joint(f"{S}_hand_root").id]
        data.qpos[a : a + 3], data.qpos[a + 3 : a + 7] = c["base_pos"][t, k], c["base_quat"][t, k]
    names = [str(n) for n in c["joint_names"]]
    data.qpos[[model.jnt_qposadr[model.joint(n).id] for n in names]] = c["joint_pos"][t]
    src = [str(n) for n in c["prop_names"]]
    for n in props:
        j = src.index(n)
        mid = model.body_mocapid[model.body(n).id]
        data.mocap_pos[mid], data.mocap_quat[mid] = c["prop_pos"][t, j], c["prop_quat"][t, j]
    mujoco.mj_forward(model, data)
    for g in range(model.ngeom):
        if model.body(model.geom_bodyid[g]).name in props:
            model.geom_rgba[g, 3] = 0.35

    # Distance of points to the held prop's mesh surface: nearest vertex of its (dense) collision meshes is enough here.
    def prop_points(name: str) -> np.ndarray:
        pts = []
        b = model.body(name).id
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == b and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                m = model.geom_dataid[g]
                v = model.mesh_vert[model.mesh_vertadr[m] : model.mesh_vertadr[m] + model.mesh_vertnum[m]]
                R = data.geom_xmat[g].reshape(3, 3)
                pts.append(v @ R.T + data.geom_xpos[g])
        return np.concatenate(pts)

    from scipy.spatial import cKDTree

    rend = mujoco.Renderer(model, tile, tile)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = small = ImageFont.load_default()
    tiles = []
    for S, spec, scores in (("L", left_shift, left_scores), ("R", right_shift, right_scores)):
        sh = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in spec.split(",") if kv}
        load = taxel_load(c[f"pressure_{'left' if S == 'L' else 'right'}"], c["pressure_times"], c["times"][t : t + 1])[0]
        load = np.where(load > 3.0, load, 0.0)
        rel = load / load.max() if load.max() > 0 else load
        rgba = np.tile([0.55, 0.55, 0.55, 1.0], (129, 1))
        on = rel > 0.02
        rgba[on, :3] = np.stack([np.ones(on.sum()), 0.9 * (1 - rel[on]), 0.1 * (1 - rel[on])], -1)
        tree = cKDTree(prop_points(HELD[S]))
        hand = model.body(f"{S}_hand").id
        lay0 = glove_layout(model, S)
        n_local = lay0.normal[lay0.body == hand].mean(0)
        n = data.xmat[hand].reshape(3, 3) @ (n_local / np.linalg.norm(n_local))
        for title, adjust in (("fitted layout", fitted_adjust()), (f"shifted: {spec} mm", shifted(sh))):
            pos, nrm = world_taxels(data, taxel_layout(model, S, adjust, roll=True))
            near = tree.query(pos)[0] < 0.005
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = pos.mean(0)
            cam.distance = 0.30
            cam.azimuth, cam.elevation = np.degrees(np.arctan2(-n[1], -n[0])), np.degrees(np.arcsin(np.clip(-n[2], -0.95, 0.95)))
            rend.update_scene(data, cam)
            p = pos + 0.0015 * nrm
            for i in range(129):
                if near[i]:
                    rch.add_dots(rend.scene, p[i : i + 1], 0.0042, np.array([[0.0, 0.0, 0.0, 1.0]]))
                rch.add_dots(rend.scene, p[i : i + 1], 0.0030, rgba[i : i + 1])
            im = Image.fromarray(rend.render().copy())
            share = load[near].sum() / max(load.sum(), 1e-9)
            prop = HELD[S].split("_")[-1]
            rch.label(im, (10, 10), [f"{'left' if S == 'L' else 'right'} hand ({prop}), {title}",
                                     f"glove load on points within 5 mm of the {prop}: {share:.0%}"], font, small)
            tiles.append((im, S, scores))
    W = Image.new("RGB", (2 * tile, 2 * tile + 70), (245, 245, 245))
    for k, (im, S, scores) in enumerate(tiles):
        W.paste(im, ((k % 2) * tile, (k // 2) * tile))
    d = ImageDraw.Draw(W)
    d.text((10, 2 * tile + 6), f"t = {time:.1f} s, mocap pose. Dots: glove sensor points by glove pressure (grey none, yellow to red = share of peak).",
           fill=(20, 20, 20), font=small)
    d.text((10, 2 * tile + 26), "Black ring: point within 5 mm of the held prop.", fill=(20, 20, 20), font=small)
    d.text((10, 2 * tile + 46), f"Agreement at the mocap pose (39 mid-grip frames): left {left_scores}, right {right_scores}.",
           fill=(20, 20, 20), font=small)
    out.parent.mkdir(parents=True, exist_ok=True)
    W.save(out)
    print("wrote", out)


if __name__ == "__main__":
    tyro.cli(main)

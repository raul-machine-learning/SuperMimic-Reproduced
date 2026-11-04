"""Picture of the glove taxel blocks fitted by scripts/data/fit_glove_blocks.py, on the hands.

  SM_ROBOT=capsule_hands_props MUJOCO_GL=egl uv run python scripts/media/render_glove_blocks.py --out /tmp/glove_blocks.png
  (SM_HAND_SHAPE=box: box hands; the fit script writes this picture after each fit)

Top row: each hand palm up at the clip's median finger pose, seen from the palm side. Small grey dots are the
neutral layout, colored dots the fitted layout (colored by finger, the palm yellow), with a line from each neutral
taxel to its fitted place. Bottom row: each hand on its prop (see-through) at its frame of highest glove load in the
grasp-corrected clip; the fitted taxels are colored by the glove's pressure (grey none, yellow to red = share of the
hand's peak) and ringed in black within 5 mm of the prop.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import tyro
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

from supermimic.data.glove_taxels import (BLOCK_FIT_FILE, PAD_BLOCKS, SINGLE_BLOCKS, fitted_blocks, palm_frame,
                                          taxel_layout, world_taxels)
from supermimic.data.sweep_mocap import taxel_load
from supermimic.robots import HAND_DATA, HAND_SHAPE

ROOT = Path(__file__).resolve().parents[2]
SIDES = ("L", "R")
SIDE_NAME = {"L": "left", "R": "right"}
HELD = {"L": "cleaningtools_broom", "R": "cleaningtools_dustpan"}
FINGER_RGB = {"thumb": (0.90, 0.10, 0.30), "index": (0.20, 0.70, 0.30), "middle": (0.00, 0.50, 0.80),
              "ring": (0.57, 0.12, 0.70), "pinky": (0.96, 0.51, 0.19), "palm": (0.95, 0.85, 0.10)}


def block_colors() -> np.ndarray:
    """(129, 4): each taxel's block color, single taxels a darker shade of their finger's."""
    rgba = np.ones((129, 4))
    rgba[:, :3] = FINGER_RGB["palm"]
    for name, (first, _, rows, cols) in PAD_BLOCKS.items():
        rgba[first : first + rows * cols, :3] = FINGER_RGB[name.split("_")[0]]
    for name, (i, _, _) in SINGLE_BLOCKS.items():
        rgba[i, :3] = 0.6 * np.array(FINGER_RGB[name.split("_")[0]])
    return rgba


def _rch():
    spec = importlib.util.spec_from_file_location("rch", ROOT / "scripts/media/render_capsule_hands.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def look(cam: mujoco.MjvCamera, at: np.ndarray, forward: np.ndarray, dist: float) -> None:
    f = forward / np.linalg.norm(forward)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:], cam.distance = at, dist
    cam.azimuth, cam.elevation = np.degrees(np.arctan2(f[1], f[0])), np.degrees(np.arcsin(np.clip(f[2], -0.95, 0.95)))


def add_lines(scene: mujoco.MjvScene, a: np.ndarray, b: np.ndarray, rgba=(0.1, 0.1, 0.1, 1.0), width: float = 0.0005) -> None:
    for p, q in zip(a, b):
        if scene.ngeom >= scene.maxgeom or np.linalg.norm(q - p) < 2e-4:
            continue
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, p, q)
        scene.ngeom += 1


def main(out: Path, clip: Path = HAND_DATA / "sweep_task023_hands.npz",
         grasp_clip: Path = HAND_DATA / "sweep_task023_hands_registered_closed.npz", tile: int = 700) -> None:
    """``clip``: for the palm-up views; ``grasp_clip``: the grasp-corrected clip for the views on the props."""
    rch = _rch()
    fit = json.loads(BLOCK_FIT_FILE.read_text())
    c, g = dict(np.load(clip)), dict(np.load(grasp_clip))
    model, props = rch.build_model(g, Path(str(g["mocap_dir"])), ghosts=False)
    data = mujoco.MjData(model)
    neutral = {S: taxel_layout(model, S) for S in SIDES}
    fitted = {S: taxel_layout(model, S, fitted_blocks(S)) for S in SIDES}
    names = [str(n) for n in g["joint_names"]]
    qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in names])
    root = {S: model.jnt_qposadr[model.joint(f"{S}_hand_root").id] for S in SIDES}
    mocap_of = {n: model.body_mocapid[model.body(n).id] for n in props}
    for gi in range(model.ngeom):
        if model.body(model.geom_bodyid[gi]).name in props:
            model.geom_rgba[gi, 3] = 0.35
    if "table" in [model.geom(i).name for i in range(model.ngeom)]:
        model.geom_rgba[model.geom("table").id, 3] = 0.0
    rend = mujoco.Renderer(model, tile, tile, max_geom=20000)
    cam = mujoco.MjvCamera()
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = small = ImageFont.load_default()
    colors = block_colors()
    away = np.array([0.0, 0.0, 2.0])

    def pose_alone(S: str) -> None:
        """Median finger pose, hand S palm up with its fingers along +x, the other hand and the props far away."""
        data.qpos[:] = 0
        data.qpos[qadr] = np.median(c["joint_pos"], 0)
        for s in SIDES:
            data.qpos[root[s] : root[s] + 3], data.qpos[root[s] + 3 : root[s] + 7] = [10, 10, 10], [1, 0, 0, 0]
        for n in props:
            data.mocap_pos[mocap_of[n]] = [20, 20, 20]
        mujoco.mj_forward(model, data)
        wrist, pn, _, mcp = palm_frame(model, data, S)
        x = mcp["middle"] - wrist
        x = (x - (x @ pn) * pn) / np.linalg.norm(x - (x @ pn) * pn)
        Q = np.stack([x, np.cross(pn, x), pn])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, Q.ravel())
        a = root[S]
        data.qpos[a : a + 3], data.qpos[a + 3 : a + 7] = away - Q @ (wrist - data.qpos[a : a + 3]), quat
        mujoco.mj_forward(model, data)

    def grasp_pose(S: str) -> np.ndarray:
        load = taxel_load(g[f"pressure_{SIDE_NAME[S]}"], g["pressure_times"], g["times"])
        t = int(np.argmax(load.sum(1)))
        data.qpos[:] = 0
        data.qpos[qadr] = g["joint_pos"][t]
        for k, s in enumerate(SIDES):
            data.qpos[root[s] : root[s] + 3], data.qpos[root[s] + 3 : root[s] + 7] = g["base_pos"][t, k], g["base_quat"][t, k]
        src = [str(n) for n in g["prop_names"]]
        for n in props:
            j = src.index(n)
            data.mocap_pos[mocap_of[n]], data.mocap_quat[mocap_of[n]] = g["prop_pos"][t, j], g["prop_quat"][t, j]
            if n != HELD[S]:
                data.mocap_pos[mocap_of[n]] = [20, 20, 20]
        mujoco.mj_forward(model, data)
        return load[t], g["times"][t] - g["times"][0]

    def prop_points(name: str) -> np.ndarray:
        pts, b = [], model.body(name).id
        for gi in range(model.ngeom):
            if model.geom_bodyid[gi] == b and model.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH:
                m = model.geom_dataid[gi]
                v = model.mesh_vert[model.mesh_vertadr[m] : model.mesh_vertadr[m] + model.mesh_vertnum[m]]
                pts.append(v @ data.geom_xmat[gi].reshape(3, 3).T + data.geom_xpos[gi])
        return np.concatenate(pts)

    def scores(S: str) -> str:
        (m0, o0), (m1, o1) = fit["before"][S]["held_out"], fit["after"][S]["held_out"]
        return f"held-out agreement {o0:.3f} -> {o1:.3f}, share MSE {m0:.3f} -> {m1:.3f}"

    tiles = []
    for S in SIDES:
        pose_alone(S)
        p0, n0 = world_taxels(data, neutral[S])
        p1, n1 = world_taxels(data, fitted[S])
        _, pn, _, _ = palm_frame(model, data, S)
        look(cam, p1.mean(0), -pn + [1e-3, 0, 0], 0.27)
        rend.update_scene(data, cam)
        add_lines(rend.scene, p0 + 0.0012 * n0, p1 + 0.0012 * n1)
        rch.add_dots(rend.scene, p0 + 0.0012 * n0, 0.0013, np.tile([0.45, 0.45, 0.45, 1.0], (129, 1)))
        rch.add_dots(rend.scene, p1 + 0.0016 * n1, 0.0026, colors)
        im = Image.fromarray(rend.render().copy())
        rch.label(im, (10, 10), [f"{SIDE_NAME[S]} hand, palm up: fitted sensor blocks",
                                 "grey: neutral layout, colored: fitted (by finger, palm yellow)", scores(S)], font, small)
        tiles.append(im)
    for S in SIDES:
        load, t = grasp_pose(S)
        p1, n1 = world_taxels(data, fitted[S])
        near = cKDTree(prop_points(HELD[S])).query(p1)[0] < 0.005
        rel = np.where(load > 3.0, load, 0.0) / max(load.max(), 1e-9)
        rgba = np.tile([0.55, 0.55, 0.55, 1.0], (129, 1))
        on = rel > 0.02
        rgba[on, :3] = np.stack([np.ones(on.sum()), 0.9 * (1 - rel[on]), 0.1 * (1 - rel[on])], -1)
        _, pn, radial, _ = palm_frame(model, data, S)
        look(cam, p1.mean(0), -pn + (0.6 if S == "L" else 1.0) * radial, 0.30)
        rend.update_scene(data, cam)
        q = p1 + 0.0015 * n1
        rch.add_dots(rend.scene, q[near], 0.0040, np.tile([0.0, 0.0, 0.0, 1.0], (int(near.sum()), 1)))
        rch.add_dots(rend.scene, q, 0.0028, rgba)
        im = Image.fromarray(rend.render().copy())
        prop = HELD[S].split("_")[-1]
        share = load[near].sum() / max(load.sum(), 1e-9)
        rch.label(im, (10, 10), [f"{SIDE_NAME[S]} hand on the {prop}, t = {t:.1f} s (peak glove load)",
                                 "glove pressure: grey none, yellow to red = share of peak",
                                 f"black ring: within 5 mm of the {prop}; {share:.0%} of the load there"], font, small)
        tiles.append(im)

    W = Image.new("RGB", (2 * tile, 2 * tile + 34), (245, 245, 245))
    for k, im in enumerate(tiles):
        W.paste(im, ((k % 2) * tile, (k // 2) * tile))
    mirror = "one mirrored set for both hands" if fit.get("mirror") else "one set per hand"
    ImageDraw.Draw(W).text((10, 2 * tile + 8), f"{HAND_SHAPE} hands, taxel blocks fitted to the glove pressure at the mocap pose "
                           f"({mirror}, {fit['frames']} frames, {fit['held_out']} held out).", fill=(20, 20, 20), font=small)
    out.parent.mkdir(parents=True, exist_ok=True)
    W.save(out)
    print("wrote", out)


if __name__ == "__main__":
    tyro.cli(main)

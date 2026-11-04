"""Picture of the glove taxels placed on the capsule hands (supermimic.data.glove_taxels).

  MUJOCO_GL=egl uv run python scripts/media/render_glove_taxels.py --out /tmp/glove_taxels.png

Six panels, with the fitted layout (scripts/data/fit_glove_taxels.py). Top: the left hand seen from its palm side
and from its thumb side, the right hand from its palm side (the clip's median finger pose, taxels colored by glove
region). Bottom, on the grasp-corrected clip at each hand's frame of highest glove load: the left hand's glove
loads, then for each hand the glove's pressed taxels against the taxels touching the prop in the replay. Each taxel
is a dot on the capsule surface with a short arrow along its outward normal.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import tyro
from PIL import Image, ImageDraw, ImageFont

from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from supermimic.data.glove_taxels import glove_layout, palm_frame, world_taxels
from supermimic.data.sweep_mocap import REGIONS, taxel_load, taxel_pressed
from supermimic.robots import HAND_DATA

ROOT = Path(__file__).resolve().parents[2]
SIDES = ("L", "R")
SIDE_NAME = {"L": "left", "R": "right"}
HELD = {"L": "cleaningtools_broom", "R": "cleaningtools_dustpan"}
AGREE = {"both": (40, 200, 70), "glove": (230, 40, 40), "sim": (40, 110, 235)}
fgt = None  # scripts/data/fit_glove_taxels.py, loaded in main
REGION_RGB = {
    "thumb_tip": (230, 25, 75), "thumb_base": (128, 0, 0), "index_tip": (60, 180, 75), "index_proximal": (0, 110, 50),
    "middle_tip": (0, 130, 200), "middle_proximal": (0, 60, 130), "ring_tip": (145, 30, 180), "ring_proximal": (80, 0, 110),
    "little_tip": (245, 130, 48), "little_proximal": (150, 70, 0), "palm_upper_index_side": (70, 240, 240),
    "palm_upper_little_side": (240, 50, 230), "palm_middle": (255, 225, 25), "palm_lower_index_side": (170, 110, 40),
    "palm_lower_little_side": (128, 128, 128),
}
REGION_OF = np.zeros(129, int)
for _r, _idx in enumerate(REGIONS.values()):
    REGION_OF[list(_idx)] = _r


def add_taxels(scene: mujoco.MjvScene, p: np.ndarray, n: np.ndarray, rgb: np.ndarray, arrow: float = 0.009) -> None:
    for i in range(len(p)):
        if scene.ngeom + 2 > scene.maxgeom:
            return
        c = np.array([*rgb[i] / 255.0, 1.0], np.float32)
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [0.0024, 0, 0], p[i] + 0.0008 * n[i],
                            np.eye(3).ravel(), c)
        scene.ngeom += 1
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), c)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.0007, p[i], p[i] + arrow * n[i])
        scene.ngeom += 1


def look(cam: mujoco.MjvCamera, at: np.ndarray, forward: np.ndarray, dist: float) -> None:
    f = forward / np.linalg.norm(forward)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:], cam.distance = at, dist
    cam.azimuth, cam.elevation = np.degrees(np.arctan2(f[1], f[0])), np.degrees(np.arcsin(np.clip(f[2], -1, 1)))


def label(im: Image.Image, xy, lines, font, small) -> None:
    dr = ImageDraw.Draw(im, "RGBA")
    x, y = xy
    w = max([dr.textlength(lines[0], font=font)] + [dr.textlength(s, font=small) for s in lines[1:]]) + 20
    dr.rounded_rectangle([x, y, x + w, y + 36 + 22 * (len(lines) - 1)], radius=8, fill=(20, 24, 32, 170))
    dr.text((x + 10, y + 8), lines[0], font=font, fill=(255, 255, 255, 255))
    for i, s in enumerate(lines[1:]):
        dr.text((x + 10, y + 34 + 22 * i), s, font=small, fill=(215, 222, 235, 255))


def _load(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(out: Path, clip: Path = HAND_DATA / "sweep_task023_hands.npz",
         grasp_clip: Path = HAND_DATA / "sweep_task023_hands_registered_closed.npz", touch: float = 0.005,
         tile_w: int = 800, tile_h: int = 600) -> None:
    """``clip``: for the hand-alone views; ``grasp_clip``: the grasp-corrected clip for the grasp views."""
    global fgt
    rch = _load("rch", ROOT / "scripts/media/render_capsule_hands.py")
    fgt = _load("fgt", ROOT / "scripts/data/fit_glove_taxels.py")

    c = dict(np.load(clip))
    model, props = rch.build_model(c, Path(str(c["mocap_dir"])), ghosts=False)
    data = mujoco.MjData(model)
    layout = {S: glove_layout(model, S) for S in SIDES}
    names = [str(n) for n in c["joint_names"]]
    qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in names])
    root = {S: model.jnt_qposadr[model.joint(f"{S}_hand_root").id] for S in SIDES}
    mocap_of = {n: model.body_mocapid[model.body(n).id] for n in props}
    rend = mujoco.Renderer(model, tile_h, tile_w)
    cam = mujoco.MjvCamera()
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    except OSError:
        font = small = ImageFont.load_default()
    region_rgb = np.array([REGION_RGB[r] for r in REGIONS])[REGION_OF]

    away = c["table_center"] + [0.0, 0.0, 2.0]

    def pose_alone(S: str) -> None:
        """Median finger pose, this hand palm up (palm normal +z) with its fingers along +x, the other far away."""
        data.qpos[:] = 0
        data.qpos[qadr] = np.median(c["joint_pos"], 0)
        for s in SIDES:
            a = root[s]
            data.qpos[a : a + 3], data.qpos[a + 3 : a + 7] = [10, 10, 10], [1, 0, 0, 0]
        for n in props:
            data.mocap_pos[mocap_of[n]] = [20, 20, 20]
        mujoco.mj_forward(model, data)
        wrist, pn, _, mcp = palm_frame(model, data, S)
        x = mcp["middle"] - wrist
        x = (x - (x @ pn) * pn) / np.linalg.norm(x - (x @ pn) * pn)
        Q = np.stack([x, np.cross(pn, x), pn])  # rows: hand axes -> world x, y, z
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, Q.ravel())
        a = root[S]
        data.qpos[a : a + 3], data.qpos[a + 3 : a + 7] = away - Q @ (wrist - data.qpos[a : a + 3]), quat
        mujoco.mj_forward(model, data)

    def shoot(S: str, forward_of, dist: float, rgb: np.ndarray) -> Image.Image:
        p, n = world_taxels(data, layout[S])
        wrist, pn, radial, _ = palm_frame(model, data, S)
        look(cam, p.mean(0), forward_of(pn, radial), dist)
        rend.update_scene(data, cam)
        add_taxels(rend.scene, p, n, rgb)
        return Image.fromarray(rend.render())

    tiles = []
    pose_alone("L")
    tiles.append((shoot("L", lambda pn, rad: -pn + [1e-3, 0, 0], 0.30, region_rgb),
                  ["Left hand, palm up, seen from above", "129 glove taxels, colored by region; arrows = outward normals"]))
    tiles.append((shoot("L", lambda pn, rad: -rad - 0.35 * pn, 0.30, region_rgb),
                  ["Left hand, palm up, seen from the thumb side", "the arrows leave the palm side of each capsule"]))
    pose_alone("R")
    tiles.append((shoot("R", lambda pn, rad: -pn + [1e-3, 0, 0], 0.30, region_rgb),
                  ["Right hand, palm up, seen from above", "same numbering, mirrored"]))

    # The grasp-corrected clip, at each hand's peak glove load.
    g = dict(np.load(grasp_clip))
    clouds = {S: fgt.prop_cloud(ROOT / "data/props", HELD[S], 0.0015) for S in SIDES}
    trees = {S: cKDTree(clouds[S][0]) for S in SIDES}

    def grasp_pose(S: str) -> tuple[int, np.ndarray, np.ndarray]:
        load = taxel_load(g[f"pressure_{SIDE_NAME[S]}"], g["pressure_times"], g["times"])
        pressed = taxel_pressed(g[f"pressure_{SIDE_NAME[S]}"], g["pressure_times"], g["times"])
        t = int(np.argmax(load.sum(1)))
        data.qpos[:] = 0
        data.qpos[qadr] = g["joint_pos"][t]
        for k, s in enumerate(SIDES):
            a = root[s]
            data.qpos[a : a + 3], data.qpos[a + 3 : a + 7] = g["base_pos"][t, k], g["base_quat"][t, k]
        for j, n in enumerate(props):
            data.mocap_pos[mocap_of[n]], data.mocap_quat[mocap_of[n]] = g["prop_pos"][t, j], g["prop_quat"][t, j]
            if n != HELD[S]:
                data.mocap_pos[mocap_of[n]] = [20, 20, 20]
        mujoco.mj_forward(model, data)
        return t, load[t], pressed[t]

    def touching(S: str, t: int) -> np.ndarray:
        j = [str(n) for n in g["prop_names"]].index(HELD[S])
        Rp = Rotation.from_quat(g["prop_quat"][t, j], scalar_first=True).as_matrix()
        p = (world_taxels(data, layout[S])[0] - g["prop_pos"][t, j]) @ Rp
        d, i = trees[S].query(p)
        s = np.einsum("ij,ij->i", p - clouds[S][0][i], clouds[S][1][i])
        return np.where(s < 0, -d, d) < touch

    view = {"L": lambda pn, rad: -pn + 0.6 * rad, "R": lambda pn, rad: -0.5 * pn + rad}

    def show_only(S: str) -> None:
        for n in props:
            model.geom_rgba[model.body_geomadr[model.body(n).id], 3] = 0.45 if n == HELD[S] else 0.0
    show_only("L")
    model.geom_rgba[model.geom("table").id, 3] = 0.0
    t, load, _ = grasp_pose("L")
    x = np.clip(load / max(load.max(), 1e-9), 0, 1)
    heat = np.stack([200 + 55 * x, 200 * (1 - x) + 30 * x, 200 * (1 - x) + 30 * x], 1)
    heat[load < 1.0] = (90, 90, 100)
    tiles.append((shoot("L", view["L"], 0.32, heat),
                  [f"Left hand on the broom, t = {g['times'][t]:.2f} s: glove", "grey unloaded, white to red = load above rest"]))
    for S in SIDES:
        show_only(S)
        t, _, pressed = grasp_pose(S)
        hit = touching(S, t)
        rgb = np.full((129, 3), 150.0)
        rgb[pressed & hit], rgb[pressed & ~hit], rgb[~pressed & hit] = AGREE["both"], AGREE["glove"], AGREE["sim"]
        f1 = 2 * (pressed & hit).sum() / max(pressed.sum() + hit.sum(), 1)
        tiles.append((shoot(S, view[S], 0.32, rgb),
                      [f"{SIDE_NAME[S].capitalize()} hand on the {HELD[S].removeprefix('cleaningtools_')}, t = {g['times'][t]:.2f} s",
                       f"glove vs replay, F1 {f1:.2f}: green both, red glove only, blue replay only"]))

    im = Image.new("RGB", (3 * tile_w, 2 * tile_h))
    for i, (tile, lines) in enumerate(tiles):
        label(tile, (14, 14), lines, font, small)
        im.paste(tile, ((i % 3) * tile_w, (i // 3) * tile_h))
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    tyro.cli(main)

"""Video of the hand/prop contact audit (scripts/data/audit_hand_prop_contact.py --out): the retargeted G1 + Inspire
replayed kinematically with the props and the table. Each Inspire link a glove region maps to is colored while
that region is loaded: green within ``touch`` of the held prop, yellow within ``near``, red beyond; unloaded links
keep their normal color. Left panel: the scene; right: close-ups of the left and right hands.

  SM_ROBOT=g1_inspire MUJOCO_GL=egl uv run python scripts/media/render_contact_audit.py \
      --retarget ~/g1-sweep-retarget/out/sweep_g1_inspire.npz --audit /tmp/contact_audit.npz --out media/contact_audit.mp4
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import tyro
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_rollout import robot_mjcf  # noqa: E402

from supermimic.robots import ROBOT  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PROP_MESHES = {"cleaningtools_broom": "cleaningtools_broom.stl", "cleaningtools_dustpan": "cleaningtools_dustpan.stl",
               "TrashBucket_chip": "trashbucket_chip.stl"}
PROP_RGBA = {"cleaningtools_broom": [0.80, 0.55, 0.30, 1], "cleaningtools_dustpan": [0.30, 0.50, 0.80, 1],
             "TrashBucket_chip": [0.55, 0.60, 0.55, 1]}
GREEN, YELLOW, RED = np.array([0.15, 0.85, 0.30, 1]), np.array([1.0, 0.80, 0.10, 1]), np.array([0.95, 0.15, 0.15, 1])


def build_model(z: dict, mocap_dir: Path) -> tuple[mujoco.MjModel, list[str]]:
    spec = mujoco.MjSpec.from_file(str(robot_mjcf()))
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    spec.visual.quality.offsamples = 8
    spec.visual.map.znear = 0.01
    spec.visual.headlight.ambient = [0.30, 0.30, 0.30]
    spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.97, 0.97, 0.98], rgb2=[0.80, 0.84, 0.90], width=256, height=256)
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.90, 0.91, 0.93], rgb2=[0.76, 0.78, 0.82], width=512, height=512)
    spec.add_material(name="grid", textures=["", "grid"], texrepeat=[2, 2], texuniform=True)
    w = spec.worldbody
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="grid", group=0, contype=0, conaffinity=0)
    w.add_light(pos=[0, 0, 5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.45, 0.45, 0.45], castshadow=True)
    scale = float(z["scale"])
    w.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, pos=z["table_center"] * scale, size=z["table_half"] * scale,
               rgba=[0.62, 0.48, 0.34, 0.6], contype=0, conaffinity=0, group=0)
    props = [str(n) for n in z["prop_names"] if (mocap_dir / PROP_MESHES[str(n)]).exists()]
    for n in props:
        spec.add_mesh(name=f"{n}_mesh", file=str(mocap_dir / PROP_MESHES[n]), scale=[0.01 * scale] * 3)
        b = w.add_body(name=n, mocap=True)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{n}_mesh", rgba=PROP_RGBA[n], contype=0, conaffinity=0, group=0)
    for a in list(spec.actuators):
        spec.delete(a)
    for s in list(spec.sensors):
        spec.delete(s)
    for k in list(spec.keys):
        spec.delete(k)
    return spec.compile(), props


def label(im: Image.Image, xy: tuple[int, int], lines: list[str], font, font_small) -> None:
    dr = ImageDraw.Draw(im, "RGBA")
    pad, x, y = 10, *xy
    w = max(dr.textlength(lines[0], font=font), *(dr.textlength(s, font=font_small) for s in lines[1:])) + 2 * pad
    h = 2 * pad + 26 + 22 * (len(lines) - 1)
    dr.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=(20, 24, 32, 160))
    dr.text((x + pad, y + pad - 2), lines[0], font=font, fill=(255, 255, 255, 255))
    for i, s in enumerate(lines[1:]):
        dr.text((x + pad, y + pad + 26 + 22 * i), s, font=font_small, fill=(215, 222, 235, 255))


def legend(im: Image.Image, xy: tuple[int, int], touch: float, near: float, font_small) -> None:
    dr = ImageDraw.Draw(im, "RGBA")
    x, y = xy
    rows = [(GREEN, f"loaded, gap < {touch * 100:.1f} cm"), (YELLOW, f"loaded, gap < {near * 100:.0f} cm"),
            (RED, f"loaded, gap > {near * 100:.0f} cm"), (None, "unloaded: normal color")]
    w = max(dr.textlength(s, font=font_small) for _, s in rows) + 44
    dr.rounded_rectangle([x, y, x + w, y + 14 + 24 * len(rows)], radius=8, fill=(20, 24, 32, 160))
    for i, (c, s) in enumerate(rows):
        yy = y + 9 + 24 * i
        if c is not None:
            dr.rectangle([x + 10, yy + 3, x + 26, yy + 17], fill=tuple(int(v * 255) for v in c))
        dr.text((x + 34, yy), s, font=font_small, fill=(230, 235, 245, 255))


def main(retarget: Path, audit: Path, out: Path, mocap_dir: Path = ROOT / "data/sweep_mocap", fps: int = 30, width: int = 1600,
         height: int = 900, touch: float = 0.005, near: float = 0.02, hand_distance: float = 0.45) -> None:
    assert ROBOT.name == "g1_inspire", "run with SM_ROBOT=g1_inspire"
    z, a = dict(np.load(retarget)), dict(np.load(audit))
    model, props = build_model(z, mocap_dir)
    data = mujoco.MjData(model)
    assert model.nq == z["qpos"].shape[1], (model.nq, z["qpos"].shape)
    names = [str(n) for n in z["prop_names"]]
    mocap_of = {n: model.body_mocapid[model.body(n).id] for n in props}
    prop_shift = float(a["prop_shift_z"])

    links = [str(n) for n in a["links"]]
    side_geoms: dict[str, list[list[int]]] = {}  # side -> per region: visual geoms of its link
    for side, s in (("left", "L"), ("right", "R")):
        side_geoms[side] = []
        for ln in links:
            bid = model.body(f"{s}_{ln}").id
            side_geoms[side].append([g for g in range(model.ngeom) if model.geom_bodyid[g] == bid and model.geom_group[g] == 2])
    base_rgba = model.geom_rgba.copy()
    hand_body = {"left": model.body("L_hand_base_link").id, "right": model.body("R_hand_base_link").id}

    W2, H2 = width // 2, height // 2
    r_main = mujoco.Renderer(model, height, W2)
    r_hand = mujoco.Renderer(model, H2, W2)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = opt.geomgroup[2] = 1
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    except OSError:
        font = font_small = ImageFont.load_default()

    times = z["times"]
    yaw = Rotation.from_quat(z["qpos"][:, 3:7], scalar_first=True).as_euler("zyx")[:, 0]
    facing = np.degrees(np.arctan2(np.sin(yaw).mean(), np.cos(yaw).mean()))
    stride = max(1, int(round((len(times) - 1) / (times[-1] - times[0]) / fps)))
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    cams = {k: mujoco.MjvCamera() for k in ("main", "left", "right")}
    targets: dict[str, np.ndarray] = {}
    for t in range(0, len(times), stride):
        data.qpos[:] = z["qpos"][t]
        for n in props:
            j = names.index(n)
            data.mocap_pos[mocap_of[n]] = z["prop_pos"][t, j] + [0, 0, prop_shift]
            data.mocap_quat[mocap_of[n]] = z["prop_quat"][t, j]
        model.geom_rgba[:] = base_rgba
        stats = {}
        for side in ("left", "right"):
            gap, loaded = a[f"{side}_gap"][t], a[f"{side}_loaded"][t]
            for r, gs in enumerate(side_geoms[side]):
                if loaded[r]:
                    c = GREEN if gap[r] < touch else YELLOW if gap[r] < near else RED
                    model.geom_rgba[gs] = c
            g = gap[loaded]
            stats[side] = (int(loaded.sum()), int((g < touch).sum()), float(np.median(g)) if g.size else np.nan)
        mujoco.mj_forward(model, data)

        focus = {"main": np.array([*data.qpos[:2], 0.85]), **{s: data.xpos[hand_body[s]].copy() for s in ("left", "right")}}
        for k, f in focus.items():
            targets[k] = f if k not in targets else 0.85 * targets[k] + 0.15 * f
        tiles = {}
        for k, rend, dist, az, el in (("main", r_main, 2.4, facing + 215, -18), ("left", r_hand, hand_distance, facing + 150, -25),
                                      ("right", r_hand, hand_distance, facing + 210, -25)):
            cam = cams[k]
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = targets[k], dist, az, el
            rend.update_scene(data, cam, opt)
            tiles[k] = rend.render().copy()
        frame = np.zeros((height, width, 3), np.uint8)
        frame[:, :W2] = tiles["main"]
        frame[:H2, W2:] = tiles["left"]
        frame[H2:2 * H2, W2:] = tiles["right"]
        im = Image.fromarray(frame)
        ImageDraw.Draw(im).line([(W2, 0), (W2, height)], fill=(30, 30, 30), width=2)
        ImageDraw.Draw(im).line([(W2, H2), (width, H2)], fill=(30, 30, 30), width=2)
        label(im, (14, 14), ["Step 2: retarget + props, glove contact audit", f"t = {times[t] - times[0]:4.1f} s   (kinematic replay, no physics)"],
              font, font_small)
        legend(im, (14, height - 120), touch, near, font_small)
        for side, y0 in (("left", 10), ("right", H2 + 10)):
            n_on, n_touch, med = stats[side]
            prop = names[int(a[f"{side}_prop"])].replace("cleaningtools_", "")
            sub = f"{n_on} regions loaded, {n_touch} touching" + (f", median gap {med * 100:.1f} cm" if n_on else "")
            label(im, (W2 + 10, y0), [f"{side} hand (holds the {prop})", sub], font, font_small)
        ff.stdin.write(np.asarray(im).tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", out)


if __name__ == "__main__":
    tyro.cli(main)

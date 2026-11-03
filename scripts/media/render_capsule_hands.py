"""Video of the capsule hands (scripts/assets/build_capsule_hands.py) with the props and the table.

  MUJOCO_GL=egl uv run python scripts/media/render_capsule_hands.py --out media/capsule_hands_replay.mp4

Without ``--rollout`` it replays the mocap clip kinematically (wrist poses, hinge angles, prop poses). With
``--rollout`` it plays a physics rollout stored on the same keys (``times``, ``base_pos``, ``base_quat``,
``joint_pos``, ``prop_pos``, ``prop_quat``, optional ``contact`` (T, 2, n_links) bool per hand link). The mocap hand
joints are drawn as small dark dots, the mocap props as translucent ghosts when the props are simulated.
Left panel: the scene; right: close-ups of the left and right hands.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import tyro
from PIL import Image, ImageDraw, ImageFont

from supermimic.data.sweep_mocap import PROP_MESHES

ROOT = Path(__file__).resolve().parents[2]
HANDS_XML = ROOT / "src/supermimic/assets/capsule_hands/capsule_hands.xml"
STATE_RGBA = {1: np.array([0.15, 0.85, 0.30, 1.0]), 2: np.array([1.0, 0.80, 0.10, 1.0]), 3: np.array([0.95, 0.15, 0.15, 1.0])}
LEGENDS = {
    "audit": [(1, "touching the held prop (gap < 5 mm)"), (2, "glove loaded, gap < 2 cm"), (3, "glove loaded, gap > 2 cm")],
    "physics": [(1, "in contact with the held prop"), (3, "glove loaded, no contact")],
}
PROP_RGBA = {"cleaningtools_broom": [0.80, 0.55, 0.30, 1], "cleaningtools_dustpan": [0.30, 0.50, 0.80, 1],
             "TrashBucket_chip": [0.55, 0.60, 0.55, 1]}


def label(im: Image.Image, xy: tuple[int, int], lines: list[str], font, font_small) -> None:
    dr = ImageDraw.Draw(im, "RGBA")
    pad, x, y = 10, *xy
    w = max([dr.textlength(lines[0], font=font)] + [dr.textlength(s, font=font_small) for s in lines[1:]]) + 2 * pad
    h = 2 * pad + 26 + 22 * (len(lines) - 1)
    dr.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=(20, 24, 32, 160))
    dr.text((x + pad, y + pad - 2), lines[0], font=font, fill=(255, 255, 255, 255))
    for i, s in enumerate(lines[1:]):
        dr.text((x + pad, y + pad + 26 + 22 * i), s, font=font_small, fill=(215, 222, 235, 255))


def build_model(clip: dict, mocap_dir: Path, ghosts: bool) -> tuple[mujoco.MjModel, list[str]]:
    spec = mujoco.MjSpec.from_file(str(HANDS_XML))
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    spec.visual.quality.offsamples = 8
    spec.visual.map.znear = 0.005
    spec.visual.headlight.ambient = [0.30, 0.30, 0.30]
    spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.90, 0.91, 0.93], rgb2=[0.76, 0.78, 0.82], width=512, height=512)
    spec.add_material(name="grid", textures=["", "grid"], texrepeat=[2, 2], texuniform=True)
    w = spec.worldbody
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="grid", contype=0, conaffinity=0)
    w.add_light(pos=[0, 0, 5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.45, 0.45, 0.45], castshadow=True)
    w.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, pos=clip["table_center"], size=clip["table_half"],
               rgba=[0.62, 0.48, 0.34, 1.0], contype=0, conaffinity=0)
    props = [str(n) for n in clip["prop_names"]]
    for n in props:
        spec.add_mesh(name=f"{n}_mesh", file=str(mocap_dir / PROP_MESHES[n]), scale=[0.01] * 3)
        for suffix, alpha in (("", 1.0), ("_ghost", 0.25)) if ghosts else (("", 1.0),):
            b = w.add_body(name=n + suffix, mocap=True)
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{n}_mesh", rgba=[*PROP_RGBA[n][:3], alpha], contype=0, conaffinity=0)
    for a in list(spec.actuators):
        spec.delete(a)
    return spec.compile(), props


def draw_legend(im: Image.Image, xy: tuple[int, int], rows: list[tuple[int, str]], font_small) -> None:
    dr = ImageDraw.Draw(im, "RGBA")
    x, y = xy
    w = max(dr.textlength(s, font=font_small) for _, s in rows) + 44
    dr.rounded_rectangle([x, y, x + w, y + 14 + 24 * len(rows)], radius=8, fill=(20, 24, 32, 160))
    for i, (st, s) in enumerate(rows):
        yy = y + 9 + 24 * i
        dr.rectangle([x + 10, yy + 3, x + 26, yy + 17], fill=tuple(int(v * 255) for v in STATE_RGBA[st]))
        dr.text((x + 34, yy), s, font=font_small, fill=(230, 235, 245, 255))


def add_dots(scene: mujoco.MjvScene, pts: np.ndarray, radius: float = 0.0035) -> None:
    for p in pts:
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [radius, 0, 0], p, np.eye(3).ravel(),
                            np.array([0.10, 0.10, 0.15, 1.0], np.float32))
        scene.ngeom += 1


def main(out: Path, clip: Path = ROOT / "data/hands/sweep_task023_hands.npz", rollout: Path | None = None, mocap_dir: Path | None = None,
         title: str = "Capsule hands: mocap replay", subtitle: str = "kinematic replay, no physics", fps: int = 30,
         width: int = 1600, height: int = 900, hand_distance: float = 0.42, view_yaw: float = 35.0, ghosts: bool | None = None,
         legend: str | None = None) -> None:
    """``ghosts``: draw the mocap props translucent (default: for physics rollouts); ``legend``: audit | physics
    (default: from the rollout's keys)."""
    c = dict(np.load(clip))
    r = dict(np.load(rollout)) if rollout else c
    mocap_dir = mocap_dir or Path(str(c["mocap_dir"]))
    if ghosts is None:
        ghosts = rollout is not None and "gap" not in r
    model, props = build_model(c, mocap_dir, ghosts=ghosts)
    data = mujoco.MjData(model)
    sides = ("L", "R")
    root_adr = [model.jnt_qposadr[model.joint(f"{S}_hand_root").id] for S in sides]
    names = [str(n) for n in c["joint_names"]]
    qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in names])
    mocap_of = {n: model.body_mocapid[model.body(n).id] for n in props}
    ghost_of = {n: model.body_mocapid[model.body(n + "_ghost").id] for n in props} if ghosts else {}

    hand_bodies = {S: [b for b in range(model.nbody) if model.body(b).name.startswith(f"{S}_")] for S in sides}
    state_geoms = {}
    if "body_state" in r:
        for S in sides:
            ids = [model.body(str(n)).id for n in r[f"body_names_{S}"]]
            state_geoms[S] = [[g for g in range(model.ngeom) if model.geom_bodyid[g] == b] for b in ids]
    base_rgba = model.geom_rgba.copy()
    legend_rows = LEGENDS.get(legend or ("audit" if "gap" in r else "physics" if "body_state" in r else ""), [])

    W2, H2 = width // 2, height // 2
    r_main = mujoco.Renderer(model, height, W2)
    r_hand = mujoco.Renderer(model, H2, W2)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    except OSError:
        font = font_small = ImageFont.load_default()

    # Look from the actor's side of the table toward it, turned by view_yaw for a three-quarter view.
    d = c["base_pos"][:, :, :2].mean((0, 1)) - c["table_center"][:2]
    facing = np.degrees(np.arctan2(-d[1], -d[0]))
    times = r["times"]
    c_times = c["times"]
    stride = max(1, round((len(times) - 1) / (times[-1] - times[0]) / fps))
    out.parent.mkdir(parents=True, exist_ok=True)
    # WebM (VP9) plays in the editor's built-in preview, which has no H.264 decoder.
    codec = (["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32", "-row-mt", "1", "-deadline", "good", "-cpu-used", "4"] if out.suffix == ".webm"
             else ["-c:v", "libx264", "-crf", "18", "-movflags", "+faststart"])
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
                           "-i", "-", *codec, "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    cams = {k: mujoco.MjvCamera() for k in ("main", "L", "R")}
    targets: dict[str, np.ndarray] = {}
    for t in range(0, len(times), stride):
        tc = min(int(np.searchsorted(c_times, times[t] - 1e-6)), len(c_times) - 1)
        for k, a in enumerate(root_adr):
            data.qpos[a : a + 3], data.qpos[a + 3 : a + 7] = r["base_pos"][t, k], r["base_quat"][t, k]
        data.qpos[qadr] = r["joint_pos"][t]
        for j, n in enumerate(props):
            data.mocap_pos[mocap_of[n]], data.mocap_quat[mocap_of[n]] = r["prop_pos"][t, j], r["prop_quat"][t, j]
            if ghost_of:
                data.mocap_pos[ghost_of[n]], data.mocap_quat[ghost_of[n]] = c["prop_pos"][tc, j], c["prop_quat"][tc, j]
        model.geom_rgba[:] = base_rgba
        if state_geoms:
            for k, S in enumerate(sides):
                for i, gs in enumerate(state_geoms[S]):
                    st = int(r["body_state"][t, k, i])
                    if st:
                        model.geom_rgba[gs] = STATE_RGBA[st]
        mujoco.mj_forward(model, data)

        focus = {"main": c["base_pos"][tc].mean(0) * [1, 1, 0] + [0, 0, c["table_center"][2] * 2 + 0.05],
                 **{S: data.xpos[hand_bodies[S][0]].copy() for S in sides}}
        for k, f in focus.items():
            targets[k] = f if k not in targets else 0.85 * targets[k] + 0.15 * f
        tiles = {}
        for k, rend, dist, az, el in (("main", r_main, 1.6, facing + view_yaw, -28), ("L", r_hand, hand_distance, facing - 40, -30),
                                      ("R", r_hand, hand_distance, facing + 40, -30)):
            cam = cams[k]
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = targets[k], dist, az, el
            rend.update_scene(data, cam)
            add_dots(rend.scene, c["human_hand_pos"][tc])
            tiles[k] = rend.render().copy()
        frame = np.zeros((height, width, 3), np.uint8)
        frame[:, :W2] = tiles["main"]
        frame[:H2, W2:] = tiles["L"]
        frame[H2 : 2 * H2, W2:] = tiles["R"]
        im = Image.fromarray(frame)
        ImageDraw.Draw(im).line([(W2, 0), (W2, height)], fill=(30, 30, 30), width=2)
        ImageDraw.Draw(im).line([(W2, H2), (width, H2)], fill=(30, 30, 30), width=2)
        label(im, (14, 14), [title, f"t = {times[t] - times[0]:4.1f} s   ({subtitle})", "dark dots: mocap hand joints"]
              + (["translucent props: mocap poses"] if ghost_of else []), font, font_small)
        if legend_rows:
            draw_legend(im, (14, height - 20 - 24 * len(legend_rows)), legend_rows, font_small)
        for S, y0 in (("L", 10), ("R", H2 + 10)):
            label(im, (W2 + 10, y0), [f"{'left' if S == 'L' else 'right'} hand", "21 hinges, capsules from the mocap bones"], font, font_small)
        ff.stdin.write(np.asarray(im).tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", out)


if __name__ == "__main__":
    tyro.cli(main)

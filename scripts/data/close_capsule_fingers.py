"""Close the capsule-hand fingers onto the held prop where the glove says they press on it.

  uv run python scripts/data/close_capsule_fingers.py

The mocap's fingers barely curl (most middle and end joints stay within +-10 deg) and its hands sit 1-4 cm off the
handles, so replaying them in physics closes on air. On every frame where a finger's glove regions are loaded
(dilated by ``dilate_s``), this curls that finger along its ``CLOSE_JOINTS`` (in the ratios of ``CLOSE_RATIO``, within
the joint ranges) until it first touches the held prop, by at most ``max_close`` rad; a finger that cannot reach the
prop is left as recorded. The closure is smoothed in time. Wrist poses, props and the glove data are untouched.

Writes the clip with the closed hinge angles as ``joint_pos`` (the recorded ones kept as ``joint_pos_mocap``) and the
closure per finger as ``closure`` (T, 2, 5) rad.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import tyro
from scipy.ndimage import binary_dilation, uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_capsule_hands import FINGERS, HELD_CANDIDATES, REGION_BODIES, SIDE_NAME, SIDES, Hands, Reference, build, kinematic_audit, loaded_regions, region_finger

from supermimic.data.glove_taxels import flexion_alignment
from supermimic.data.sweep_mocap import REGIONS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets"))
from build_capsule_hands import median_pose
from supermimic.robots import HAND_DATA

ROOT = Path(__file__).resolve().parents[2]
# share of the closure each hinge takes, by its position along the finger; the sign of each hinge is the one that
# moves the fingertip toward the palm (supermimic.data.glove_taxels.flexion_alignment)
CLOSE_RATIO = {"thumb": {1: 0.7, 2: 1.0, 3: 0.7}, "finger": {1: 1.0, 2: 1.0, 3: 0.7}}
# The thumb's base flexion hinge swings the thumb across the palm, past a handle in the hand; its sideways hinge is
# the one that brings the thumb pad onto the handle.
CLOSE_JOINTS = {"thumb": ((1, "spread"), (2, "flex"), (3, "flex")),
                **{f: ((1, "flex"), (2, "flex"), (3, "flex")) for f in ("index", "middle", "ring", "pinky")}}


def main(clip: Path = HAND_DATA / "sweep_task023_hands.npz", out: Path | None = None, props_dir: Path = ROOT / "data/props",
         max_close: float = 1.2, step: float = 0.04, contact: float = 0.001, dilate_s: float = 0.1, smooth_s: float = 0.1,
         ignore: tuple[str, ...] = ()) -> None:
    """``contact``: gap (m) that counts as touching; ``ignore``: fingers never closed."""
    out = out or clip.with_name(clip.stem + "_closed.npz")
    c = dict(np.load(clip))
    audit = kinematic_audit(c, props_dir, 0.005, 0.02)
    held = audit["held"]
    model = build(c, True, 0.002, {n: 1.0 for n in HELD_CANDIDATES}, props_dir, 1.0, 1.0, 0.005)
    data = mujoco.MjData(model)
    ref = Reference(c)
    names = [str(n) for n in c["joint_names"]]
    hands = Hands(model, names)
    T = len(ref.t)
    fps = (T - 1) / (ref.t[-1] - ref.t[0])
    loaded = loaded_regions(c)
    regions = list(REGIONS)
    prop_qadr = {n: model.jnt_qposadr[model.joint(f"{n}_free").id] for n in HELD_CANDIDATES}
    prop_geoms = {n: [g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body(n).id] for n in HELD_CANDIDATES}

    fingers = {}  # (side, finger) -> (closing hinge columns, signed ratios, finger geoms)
    pose = median_pose(model, {n: c["joint_pos"][:, i] for i, n in enumerate(names)})
    for S in SIDES:
        for f in FINGERS:
            cols = [names.index(f"{S}_{f}{k}_{kind}") for k, kind in CLOSE_JOINTS[f]]
            ratio = np.array([CLOSE_RATIO["thumb" if f == "thumb" else "finger"][k] for k, _ in CLOSE_JOINTS[f]])
            ratio *= [np.sign(flexion_alignment(model, pose, names[i])) for i in cols]
            geoms = [g for k in (1, 2, 3) for g in hands.geoms[model.body(f"{S}_{f}{k}").id]]
            fingers[(S, f)] = (np.array(cols), ratio, geoms)
    lo, hi = model.jnt_range[[model.joint(n).id for n in names]].T

    def finger_gap(geoms: list[int], prop: str, reach: float = 0.03) -> float:
        best, fromto = reach, np.zeros(6)
        for g1 in geoms:
            for g2 in prop_geoms[prop]:
                if np.linalg.norm(data.geom_xpos[g1] - data.geom_xpos[g2]) - model.geom_rbound[g1] - model.geom_rbound[g2] < best:
                    best = min(best, mujoco.mj_geomDistance(model, data, g1, g2, reach, fromto))
        return best

    want = {}
    for k, S in enumerate(SIDES):
        for f in FINGERS:
            cols = [r for r, reg in enumerate(regions) if region_finger(reg) == f]
            on = loaded[S][:, cols].any(1)
            want[(S, f)] = binary_dilation(on, iterations=max(1, round(dilate_s * fps))) if f not in ignore else np.zeros(T, bool)

    closure = np.zeros((T, 2, len(FINGERS)))
    reached = np.zeros((T, 2, len(FINGERS)), bool)
    levels = np.arange(0.0, max_close + 1e-9, step)
    for t in range(T):
        hands.set_pose(data, ref, ref.t[t], with_vel=False)
        for n in HELD_CANDIDATES:
            p, q = ref.prop(ref.t[t], ref.props.index(n))
            data.qpos[prop_qadr[n] : prop_qadr[n] + 3], data.qpos[prop_qadr[n] + 3 : prop_qadr[n] + 7] = p, q
        q_rec = data.qpos[hands.jq].copy()
        for k, S in enumerate(SIDES):
            prop = HELD_CANDIDATES[held[S]]
            for fi, f in enumerate(FINGERS):
                if not want[(S, f)][t]:
                    continue
                cols, ratio, geoms = fingers[(S, f)]
                for cl in levels:
                    data.qpos[hands.jq[cols]] = np.clip(q_rec[cols] + cl * ratio, lo[cols], hi[cols])
                    mujoco.mj_kinematics(model, data)
                    if finger_gap(geoms, prop) <= contact:
                        closure[t, k, fi], reached[t, k, fi] = cl, True
                        break
                data.qpos[hands.jq[cols]] = q_rec[cols]
    closure = uniform_filter1d(closure, max(1, round(smooth_s * fps)), axis=0)

    joint_pos = c["joint_pos"].copy()
    for k, S in enumerate(SIDES):
        for fi, f in enumerate(FINGERS):
            cols, ratio, _ = fingers[(S, f)]
            joint_pos[:, cols] = np.clip(joint_pos[:, cols] + closure[:, k, fi, None] * ratio, lo[cols], hi[cols])
    np.savez_compressed(out, **{**c, "joint_pos": joint_pos, "joint_pos_mocap": c["joint_pos"], "closure": closure, "closure_fingers": np.array(FINGERS)})
    print(f"wrote {out}")
    for k, S in enumerate(SIDES):
        prop = HELD_CANDIDATES[held[S]].removeprefix("cleaningtools_")
        print(f"  {SIDE_NAME[S]} hand on the {prop}:")
        for fi, f in enumerate(FINGERS):
            w = want[(S, f)]
            if not w.any():
                print(f"    {f:7s} never loaded")
                continue
            r = reached[w, k, fi]
            cl = closure[w, k, fi]
            print(f"    {f:7s} loaded {w.mean():4.0%} of frames; reaches the prop on {r.mean():4.0%} of them; "
                  f"closure median {np.degrees(np.median(cl[r])) if r.any() else 0:4.0f} deg, max {np.degrees(cl.max()):4.0f} deg")


if __name__ == "__main__":
    tyro.cli(main)

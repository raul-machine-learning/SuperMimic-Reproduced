"""Register each mocap hand to the prop it holds: one fixed shift of the wrist in the prop's frame.

  uv run python scripts/data/register_hands_to_props.py

While a prop is lifted, the recorded wrist pose in the prop's frame is rigid (std ~3 mm, ~2.5 deg), yet the capsule
hand sits 1-4 cm off the handle where the glove says it presses: a constant offset between the hand and prop
trackers (and the capsules being thinner than a real palm). Per hand, this finds the translation ``d`` in the held
prop's frame that brings the loaded glove regions onto the prop surface (squared gap) without pushing into it
(squared penetration beyond ``allow_pen``), over the frames the hand grasps the prop. It is applied on those frames
only, blended in and out over ``blend_s``: wrist position += R_prop(t) d. Orientations, fingers and props are
untouched; the recorded wrist positions are kept as ``base_pos_mocap``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import tyro
from scipy.ndimage import binary_closing, binary_dilation, uniform_filter1d
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_capsule_hands import HELD_CANDIDATES, REGION_BODIES, SIDE_NAME, SIDES, Hands, Reference, build, kinematic_audit

from supermimic.data.sweep_mocap import REGIONS

ROOT = Path(__file__).resolve().parents[2]


def main(clip: Path = ROOT / "data/hands/sweep_task023_hands.npz", out: Path | None = None, props_dir: Path = ROOT / "data/props",
         grasp_gap: float = 0.03, allow_pen: float = 0.002, pen_weight: float = 20.0, max_shift: float = 0.05, n_frames: int = 60,
         blend_s: float = 0.25) -> None:
    """``grasp_gap``: a hand grasps its prop on frames with a loaded glove region within this gap of it."""
    out = out or clip.with_name(clip.stem + "_registered.npz")
    c = dict(np.load(clip))
    audit = kinematic_audit(c, props_dir, 0.005, 0.02)
    held = audit["held"]
    model = build(c, True, 0.002, {n: 1.0 for n in HELD_CANDIDATES}, props_dir, 1.0, 1.0, 0.005)
    data = mujoco.MjData(model)
    ref = Reference(c)
    hands = Hands(model, [str(n) for n in c["joint_names"]])
    T = len(ref.t)
    fps = (T - 1) / (ref.t[-1] - ref.t[0])
    regions = list(REGIONS)
    prop_qadr = {n: model.jnt_qposadr[model.joint(f"{n}_free").id] for n in HELD_CANDIDATES}
    prop_geoms = {n: [g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body(n).id] for n in HELD_CANDIDATES}
    base_pos = c["base_pos"].copy()
    fromto = np.zeros(6)

    def gap(geoms: list[int], prop: str, reach: float = 0.06) -> float:
        best = reach
        for g1 in geoms:
            for g2 in prop_geoms[prop]:
                if np.linalg.norm(data.geom_xpos[g1] - data.geom_xpos[g2]) - model.geom_rbound[g1] - model.geom_rbound[g2] < best:
                    best = min(best, mujoco.mj_geomDistance(model, data, g1, g2, reach, fromto))
        return best

    for k, S in enumerate(SIDES):
        prop = HELD_CANDIDATES[held[S]]
        j = ref.props.index(prop)
        names = audit["body_names"][S]
        g_all = audit["gap"][:, k, :, held[S]]
        on = audit["loaded"][S]
        near = np.zeros(T, bool)
        for r, reg in enumerate(regions):
            idx = [names.index(f"{S}_{b}") for b in REGION_BODIES[reg]]
            near |= on[:, r] & (g_all[:, idx].min(1) < grasp_gap)
        grasp = binary_closing(near, iterations=round(0.2 * fps)) & binary_dilation(near, iterations=1)
        frames = np.flatnonzero(grasp)
        sample = frames[np.linspace(0, len(frames) - 1, min(n_frames, len(frames))).astype(int)]
        R_prop = Rotation.from_quat(c["prop_quat"][:, j], scalar_first=True)
        body_geoms = {b: hands.geoms[b] for b in hands.bodies[S]}

        def pose(t: int, d: np.ndarray) -> None:
            hands.set_pose(data, ref, ref.t[t], with_vel=False)
            a = hands.root_q[k]
            data.qpos[a : a + 3] = c["base_pos"][t, k] + R_prop[t].apply(d)
            data.qpos[prop_qadr[prop] : prop_qadr[prop] + 3], data.qpos[prop_qadr[prop] + 3 : prop_qadr[prop] + 7] = c["prop_pos"][t, j], c["prop_quat"][t, j]
            mujoco.mj_kinematics(model, data)

        def cost(d: np.ndarray) -> float:
            total = 0.0
            for t in sample:
                pose(t, d)
                g_body = {b: gap(gs, prop) for b, gs in body_geoms.items()}
                for r, reg in enumerate(regions):
                    if on[t, r]:
                        gr = min(g_body[model.body(f"{S}_{b}").id] for b in REGION_BODIES[reg])
                        total += max(gr, 0.0) ** 2
                total += pen_weight * sum(max(-gv - allow_pen, 0.0) ** 2 for gv in g_body.values())
            return total / len(sample)

        def stats(d: np.ndarray) -> tuple[float, float, float]:
            gaps, pens = [], []
            for t in sample:
                pose(t, d)
                g_body = {b: gap(gs, prop) for b, gs in body_geoms.items()}
                for r, reg in enumerate(regions):
                    if on[t, r]:
                        gaps.append(min(g_body[model.body(f"{S}_{b}").id] for b in REGION_BODIES[reg]))
                pens.append(-min(g_body.values()))
            gaps = np.array(gaps)
            return float(np.median(gaps)), float(np.mean(gaps < 0.005)), float(np.max(pens))

        res = minimize(cost, np.zeros(3), method="Nelder-Mead", options=dict(initial_simplex=np.vstack([np.zeros(3), np.eye(3) * 0.01]),
                                                                              xatol=5e-4, fatol=1e-8, maxiter=400))
        d = np.clip(res.x, -max_shift, max_shift)
        before, after = stats(np.zeros(3)), stats(d)
        w = uniform_filter1d(grasp.astype(float), max(1, round(blend_s * fps)))
        base_pos[:, k] = c["base_pos"][:, k] + w[:, None] * R_prop.apply(d)
        print(f"{SIDE_NAME[S]} hand on the {prop.removeprefix('cleaningtools_')}: grasps {grasp.mean():.0%} of the clip; "
              f"shift in the prop frame {np.round(d * 1000, 1)} mm (|d| {np.linalg.norm(d) * 1000:.1f} mm)")
        print(f"  loaded-region gap median {before[0] * 1000:5.1f} -> {after[0] * 1000:5.1f} mm, touching (< 5 mm) {before[1]:4.0%} -> {after[1]:4.0%}, "
              f"deepest penetration {before[2] * 1000:4.1f} -> {after[2] * 1000:4.1f} mm")
    np.savez_compressed(out, **{**c, "base_pos": base_pos, "base_pos_mocap": c["base_pos"]})
    print(f"wrote {out}")


if __name__ == "__main__":
    tyro.cli(main)

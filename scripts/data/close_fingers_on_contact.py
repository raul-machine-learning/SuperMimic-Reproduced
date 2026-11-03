"""Close the retargeted Inspire fingers onto the props wherever the glove reports contact.

  SM_ROBOT=g1_inspire uv run python scripts/data/close_fingers_on_contact.py \
      --retarget ~/g1-sweep-retarget/out/sweep_g1_inspire.npz --out ~/g1-sweep-retarget/out/sweep_g1_inspire_closed.npz

The retarget maps finger angles from the human, and those fingers stop centimeters short of the handles even where the
glove is loaded (scripts/data/audit_hand_prop_contact.py). Per frame and per finger (index to little; the thumb
already touches), this sweeps the finger's one actuated angle (the intermediate follows through its mimic coupling) on
SuperMimic's collision geometry against the props' convex pieces (scripts/assets/build_props.py) and:

  * glove region of the finger loaded: close it from the reference angle until the loaded link (tip region ->
    intermediate, proximal region -> proximal) touches a prop, never driving either link deeper than ``max_pen``
    into a prop; out of reach -> keep the reference;
  * not loaded: only open the finger if it is deeper than ``max_pen`` inside a prop.

The angle change is low-passed (``smooth_s``) so the fingers close at a plausible speed. Writes the retarget npz with
the new ``qpos`` plus ``finger_closure`` (T, fingers) in rad and ``finger_closure_joints``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import tyro
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[2]
RETARGET_ROOT = Path.home() / "g1-sweep-retarget"
FINGERS = (("index", "index"), ("middle", "middle"), ("ring", "ring"), ("little", "pinky"))  # glove name, Inspire name
FINGER_BIT = 2


def build_model(z: dict, props_dir: Path, margin: float):
    from supermimic.robots import ROBOT

    spec = ROBOT.spec_fn()
    for g in spec.geoms:
        g.contype, g.conaffinity = 0, 0
    scale = float(z["scale"])
    names = [str(n) for n in z["prop_names"]]
    for n in names:
        b = spec.worldbody.add_body(name=n, mocap=True)
        for i, f in enumerate(sorted((props_dir / n).glob("collision_*.obj"))):
            spec.add_mesh(name=f"{n}_c{i:02d}", file=str(f.resolve()), scale=[scale] * 3)
            b.add_geom(name=f"{n}_c{i:02d}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{n}_c{i:02d}", contype=0,
                       conaffinity=FINGER_BIT, margin=margin, group=3)
    links = {}
    for side, s in (("left", "L"), ("right", "R")):
        for _, f in FINGERS:
            for part in ("proximal", "intermediate"):
                g = spec.geom(f"{s}_{f}_{part}_collision")
                g.contype, g.conaffinity, g.margin = FINGER_BIT, 0, margin
                links[(side, f, part)] = g.name
    model = spec.compile()
    return model, names, {k: model.geom(v).id for k, v in links.items()}


def glove_region_loaded(z: dict, retarget_root: Path, rest: tuple[float, float]) -> dict[str, np.ndarray]:
    """side -> (T, regions) bool on the retarget's time grid: region load above 15% of its clip maximum (and 3 raw units)."""
    sys.path.insert(0, str(retarget_root))
    from sweep_retarget.pressure import contact_load, region_load

    pt, times = z["pressure_times"], z["times"]
    out = {}
    for side in ("left", "right"):
        load = region_load(contact_load(z[f"pressure_{side}"], pt, rest))
        load = np.stack([np.interp(times, pt, load[:, r]) for r in range(load.shape[1])], 1)
        out[side] = load > np.maximum(0.15 * load.max(0), 3.0)
    return out


def main(retarget: Path, out: Path, props_dir: Path = ROOT / "data/props",
         retarget_root: Path = RETARGET_ROOT, rest: tuple[float, float] = (0.0, 2.0),
         step: float = 0.02, max_pen: float = 0.002, margin: float = 0.02, smooth_s: float = 0.1) -> None:
    from supermimic.robots import ROBOT

    assert ROBOT.name == "g1_inspire", "run with SM_ROBOT=g1_inspire"
    sys.path.insert(0, str(retarget_root))
    from sweep_retarget.pressure import REGIONS

    z = dict(np.load(retarget))
    assert bool(z.get("props_grounded", False)), "retarget predates grounded props: re-run retarget.py"
    model, prop_names, link_geom = build_model(z, props_dir, margin)
    data = mujoco.MjData(model)
    names = [str(n) for n in z["joint_names"]]
    assert tuple(names) == ROBOT.joint_names
    times, qpos = z["times"], z["qpos"].copy()
    T = len(times)
    mocap = [model.body_mocapid[model.body(n).id] for n in prop_names]
    geom_finger = {g: k for k, g in link_geom.items()}

    def qadr(j: str) -> int:
        return model.jnt_qposadr[model.joint(j).id]

    fingers = [(side, s, gf, f) for side, s in (("left", "L"), ("right", "R")) for gf, f in FINGERS]
    drv = np.array([qadr(f"{s}_{f}_proximal_joint") for _, s, _, f in fingers])
    fol = np.array([qadr(f"{s}_{f}_intermediate_joint") for _, s, _, f in fingers])
    mult = np.array([model.eq_data[next(e for e in range(model.neq) if model.eq_obj1id[e] == model.joint(f"{s}_{f}_intermediate_joint").id)][1]
                     for _, s, _, f in fingers])
    lo, hi = model.jnt_range[model.joint("L_index_proximal_joint").id]
    grid = np.arange(lo, hi + 1e-9, step)

    reg = list(REGIONS)
    region_on = glove_region_loaded(z, retarget_root, rest)
    loaded = {}
    for side in ("left", "right"):
        on = region_on[side]
        for gf, f in FINGERS:
            loaded[(side, f, "intermediate")] = on[:, reg.index(f"{gf}_tip")]
            loaded[(side, f, "proximal")] = on[:, reg.index(f"{gf}_proximal")]

    def link_dists(q: np.ndarray, t: int) -> np.ndarray:
        """(fingers, 2) signed distance of the proximal / intermediate link to the nearest prop (``margin`` if none)."""
        data.qpos[:] = q
        for j, m in enumerate(mocap):
            data.mocap_pos[m], data.mocap_quat[m] = z["prop_pos"][t, j], z["prop_quat"][t, j]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        d = np.full((len(fingers), 2), margin)
        for c in data.contact[: data.ncon]:
            for g in (c.geom1, c.geom2):
                if g in geom_finger:
                    side, f, part = geom_finger[g]
                    i = next(k for k, x in enumerate(fingers) if x[0] == side and x[3] == f)
                    p = 0 if part == "proximal" else 1
                    d[i, p] = min(d[i, p], c.dist)
        return d

    theta_ref = qpos[:, drv]
    theta = theta_ref.copy()
    unreachable = np.zeros((T, len(fingers)), bool)
    D_all = np.zeros((T, len(grid), len(fingers), 2))  # link distances over the angle grid (fingers are independent)
    for t in range(T):
        q = qpos[t].copy()
        d_ref = link_dists(q, t)
        D = D_all[t]
        for k, th in enumerate(grid):
            q[drv], q[fol] = th, mult * th
            D[k] = link_dists(q, t)
        for i, (side, s, gf, f) in enumerate(fingers):
            want = np.array([loaded[(side, f, "proximal")][t], loaded[(side, f, "intermediate")][t]])
            ok = D[:, i].min(1) >= -max_pen
            r = theta_ref[t, i]
            if d_ref[i].min() < -max_pen:
                down = np.flatnonzero(ok & (grid < r))
                if len(down):
                    theta[t, i] = grid[down[-1]]
            elif want.any() and d_ref[i][want].min() > 0.0:
                tgt = D[:, i][:, want].min(1)
                up = np.flatnonzero(grid > r)
                hit = next((k for k in up if tgt[k] <= 0.0 or not ok[k]), None)
                if hit is None:
                    unreachable[t, i] = True
                elif ok[hit]:
                    theta[t, i] = grid[hit]
                elif hit != up[0]:
                    theta[t, i] = grid[hit - 1]

    fps = (T - 1) / (times[-1] - times[0])
    delta = gaussian_filter1d(theta - theta_ref, sigma=smooth_s * fps, axis=0, mode="nearest")
    new = np.clip(theta_ref + delta, lo, hi)
    # Smoothing can push a closed finger back into the prop: open it to the nearest grid angle that is clear.
    for t in range(T):
        for i in range(len(fingers)):
            deep = min(np.interp(new[t, i], grid, D_all[t, :, i, p]) for p in (0, 1)) < -max_pen
            clear = np.flatnonzero((D_all[t, :, i].min(1) >= -max_pen) & (grid <= new[t, i]))
            if deep and len(clear):
                new[t, i] = grid[clear[-1]]
    delta = new - theta_ref
    qpos[:, drv] = new
    qpos[:, fol] = new * mult

    gaps_ref = np.array([link_dists(z["qpos"][t], t) for t in range(T)])  # (T, fingers, 2)
    gaps = np.array([link_dists(qpos[t], t) for t in range(T)])
    print(f"{retarget.name}: {T} frames. Per finger, on frames where its glove regions are loaded: closure (rad), "
          f"gap of the loaded link to the prop before -> after (median, share within 5 mm; gaps saturate at "
          f"{margin * 100:.0f} cm), deepest penetration after")
    for i, (side, s, gf, f) in enumerate(fingers):
        want = np.stack([loaded[(side, f, "proximal")], loaded[(side, f, "intermediate")]], 1)
        on = want.any(1)
        if not on.any():
            continue
        before = np.where(want, gaps_ref[:, i], np.inf).min(1)[on] * 100
        after = np.where(want, gaps[:, i], np.inf).min(1)[on] * 100
        print(f"  {side:5s} {gf:6s} loaded {on.mean():4.0%}  closure median {np.median(delta[on, i]):+.2f}  "
              f"gap {np.median(before):4.1f} -> {np.median(after):4.1f} cm  within 5 mm {np.mean(before < 0.5):4.0%} -> {np.mean(after < 0.5):4.0%}  "
              f"out of reach {unreachable[on, i].mean():4.0%}  deepest {gaps[:, i].min() * 1000:+.1f} mm")
    z["qpos"] = qpos
    z["finger_closure"] = delta
    z["finger_closure_joints"] = np.array([f"{s}_{f}_proximal_joint" for _, s, _, f in fingers])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **z)
    print(f"wrote {out}")


if __name__ == "__main__":
    tyro.cli(main)

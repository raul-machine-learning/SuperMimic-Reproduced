"""Can the capsule hands (scripts/assets/build_capsule_hands.py) track the sweep mocap and hold the props in physics?

  uv run python scripts/data/check_capsule_hands.py --out-dir /tmp/capsule_check

Three checks, on the mocap clip alone:

  audit  kinematic replay: for every glove region that is loaded, the gap between the hand capsules that carry it
         and the prop the hand holds (which tells whether the recorded grasp touches the prop where the glove says).
  air    physics, no props or table: each wrist is a free body pulled toward the recorded wrist pose by a
         force-limited PD wrench (computed torque on its 6 root dofs, gravity compensated); fingers are
         position-actuated toward the recorded hinge angles.
  hold   the same hands with the broom and dustpan as free bodies (convex pieces from scripts/assets/build_props.py,
         real scale) on a static table, once per ``squeeze``: on frames where a finger's glove regions are loaded its
         flexion targets close further by ``squeeze`` rad, since a target exactly at the surface gives no grip force.

Each run writes a rollout for scripts/media/render_capsule_hands.py. ``body_state`` per hand body: 1 touching the
held prop (audit: gap < ``touch``; physics: in contact), 2 glove-loaded with a gap < ``near``, 3 glove-loaded and
farther, 0 otherwise.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import tyro
from scipy.spatial.transform import Rotation, Slerp

from supermimic.data.sweep_mocap import REGIONS, TABLE_TOP_HEIGHT, region_loaded
from supermimic.robots import HAND_DATA, HANDS_XML

ROOT = Path(__file__).resolve().parents[2]
SIDES = ("L", "R")
SIDE_NAME = {"L": "left", "R": "right"}
HELD_CANDIDATES = ("cleaningtools_broom", "cleaningtools_dustpan")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
GLOVE_FINGER = {"thumb": "thumb", "index": "index", "middle": "middle", "ring": "ring", "little": "pinky"}
# glove region -> hand bodies (without the side prefix) whose capsules carry it
REGION_BODIES = {
    "thumb_tip": ("thumb3",), "thumb_base": ("thumb1",),
    **{f"{g}_tip": (f"{f}2", f"{f}3") for g, f in GLOVE_FINGER.items() if f != "thumb"},
    **{f"{g}_proximal": (f"{f}1",) for g, f in GLOVE_FINGER.items() if f != "thumb"},
    "palm_upper_index_side": ("index0", "middle0"), "palm_upper_little_side": ("ring0", "pinky0"),
    "palm_middle": ("hand", "middle0", "ring0"), "palm_lower_index_side": ("hand",), "palm_lower_little_side": ("hand",),
}
assert set(REGION_BODIES) == set(REGIONS)


def region_finger(region: str) -> str | None:
    head = region.split("_")[0]
    return GLOVE_FINGER.get(head)


def build(clip: dict, with_props: bool, dt: float, masses: dict[str, float], props_dir: Path, prop_friction: float,
          hand_friction: float, torsional: float) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(HANDS_XML))
    spec.option.timestep = dt
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for g in spec.geoms:
        g.friction = [hand_friction, torsional, 0.0001]
    if with_props:
        w = spec.worldbody
        w.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, pos=clip["table_center"], size=clip["table_half"], condim=3,
                   friction=[prop_friction, 0.005, 0.0001])
        for n in HELD_CANDIDATES:
            b = w.add_body(name=n)
            b.add_freejoint(name=f"{n}_free")
            for i, f in enumerate(sorted((props_dir / n).glob("collision_*.obj"))):
                spec.add_mesh(name=f"{n}_c{i:02d}", file=str(f.resolve()))
                b.add_geom(name=f"{n}_c{i:02d}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{n}_c{i:02d}", condim=3,
                           friction=[prop_friction, 0.005, 0.0001], density=1.0, group=3)
    model = spec.compile()
    if not with_props:
        return model
    # The convex pieces at density 1 give each prop's inertia shape; scale it to the requested mass in the spec, since
    # editing body_mass on a compiled model leaves its derived mass data stale.
    for n in HELD_CANDIDATES:
        b = model.body(n).id
        k = masses[n] / model.body_mass[b]
        body = spec.body(n)
        body.explicitinertial = True
        body.mass, body.ipos, body.iquat, body.inertia = masses[n], model.body_ipos[b], model.body_iquat[b], model.body_inertia[b] * k
    return spec.compile()


class Reference:
    """The clip on its own time grid, interpolated in time (positions linearly, orientations by slerp)."""

    def __init__(self, c: dict):
        self.c, self.t = c, c["times"]
        self.props = [str(n) for n in c["prop_names"]]
        self.slerp_base = [Slerp(self.t, Rotation.from_quat(c["base_quat"][:, k], scalar_first=True)) for k in range(2)]
        self.slerp_prop = [Slerp(self.t, Rotation.from_quat(c["prop_quat"][:, j], scalar_first=True)) for j in range(len(self.props))]
        self.base_vel = np.gradient(c["base_pos"], self.t, axis=0)
        self.base_omega = np.stack([self._omega(s) for s in self.slerp_base], 1)  # (T, 2, 3) world frame
        self.joint_vel = np.gradient(c["joint_pos"], self.t, axis=0)

    def _omega(self, s: Slerp) -> np.ndarray:
        h = 1e-3
        t = np.clip(self.t, self.t[0], self.t[-1] - h)
        return (s(t + h) * s(t).inv()).as_rotvec() / h

    def _lerp(self, x: np.ndarray, tq: float) -> np.ndarray:
        i = int(np.clip(np.searchsorted(self.t, tq) - 1, 0, len(self.t) - 2))
        a = float(np.clip((tq - self.t[i]) / (self.t[i + 1] - self.t[i]), 0.0, 1.0))
        return (1 - a) * x[i] + a * x[i + 1]

    def base(self, tq: float, k: int):
        tq = float(np.clip(tq, self.t[0], self.t[-1]))
        q = self.slerp_base[k](tq).as_quat(scalar_first=True)
        return self._lerp(self.c["base_pos"][:, k], tq), q, self._lerp(self.base_vel[:, k], tq), self._lerp(self.base_omega[:, k], tq)

    def joints(self, tq: float):
        return self._lerp(self.c["joint_pos"], tq), self._lerp(self.joint_vel, tq)

    def prop(self, tq: float, j: int):
        tq = float(np.clip(tq, self.t[0], self.t[-1]))
        return self._lerp(self.c["prop_pos"][:, j], tq), self.slerp_prop[j](tq).as_quat(scalar_first=True)

    def frame(self, tq: float) -> int:
        return int(np.clip(np.searchsorted(self.t, tq - 1e-9), 0, len(self.t) - 1))


class Hands:
    """Index bookkeeping of the two hands in a compiled model."""

    def __init__(self, model: mujoco.MjModel, joint_names: list[str]):
        self.m = model
        self.root_q = [model.jnt_qposadr[model.joint(f"{S}_hand_root").id] for S in SIDES]
        self.root_d = [model.jnt_dofadr[model.joint(f"{S}_hand_root").id] for S in SIDES]
        self.jq = np.array([model.jnt_qposadr[model.joint(n).id] for n in joint_names])
        self.jd = np.array([model.jnt_dofadr[model.joint(n).id] for n in joint_names])
        self.act = np.array([model.actuator(n).id for n in joint_names])
        self.bodies = {S: [b for b in range(model.nbody) if model.body(b).name.startswith(f"{S}_")] for S in SIDES}
        self.body_names = {S: [model.body(b).name for b in self.bodies[S]] for S in SIDES}
        self.body_side = {b: (k, i) for k, S in enumerate(SIDES) for i, b in enumerate(self.bodies[S])}
        self.geoms = {b: [g for g in range(model.ngeom) if model.geom_bodyid[g] == b] for S in SIDES for b in self.bodies[S]}

    def set_pose(self, data: mujoco.MjData, ref: Reference, tq: float, with_vel: bool = True) -> None:
        for k in range(2):
            p, q, v, w = ref.base(tq, k)
            a, d = self.root_q[k], self.root_d[k]
            data.qpos[a : a + 3], data.qpos[a + 3 : a + 7] = p, q
            if with_vel:
                data.qvel[d : d + 3] = v
                data.qvel[d + 3 : d + 6] = Rotation.from_quat(q, scalar_first=True).inv().apply(w)
        qj, vj = ref.joints(tq)
        data.qpos[self.jq] = qj
        if with_vel:
            data.qvel[self.jd] = vj


def body_index(hands: Hands, S: str, short: str) -> int:
    return hands.body_names[S].index(f"{S}_{short}")


def loaded_regions(c: dict) -> dict[str, np.ndarray]:
    return {S: region_loaded(c[f"pressure_{SIDE_NAME[S]}"], c["pressure_times"], c["times"]) for S in SIDES}


def kinematic_audit(c: dict, props_dir: Path, touch: float, near: float, reach: float = 0.05) -> dict:
    """Gap (m, capped at ``reach``) of every hand body to each held-prop candidate, on every mocap frame."""
    model = build(c, True, 0.002, {n: 1.0 for n in HELD_CANDIDATES}, props_dir, 1.0, 1.0, 0.005)
    data = mujoco.MjData(model)
    ref = Reference(c)
    hands = Hands(model, [str(n) for n in c["joint_names"]])
    prop_geoms = {n: [g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body(n).id] for n in HELD_CANDIDATES}
    prop_qadr = {n: model.jnt_qposadr[model.joint(f"{n}_free").id] for n in HELD_CANDIDATES}
    T, nb = len(ref.t), len(hands.bodies["L"])
    gap = np.full((T, 2, nb, len(HELD_CANDIDATES)), reach)
    fromto = np.zeros(6)
    for t in range(T):
        hands.set_pose(data, ref, ref.t[t], with_vel=False)
        for n in HELD_CANDIDATES:
            p, q = ref.prop(ref.t[t], ref.props.index(n))
            data.qpos[prop_qadr[n] : prop_qadr[n] + 3], data.qpos[prop_qadr[n] + 3 : prop_qadr[n] + 7] = p, q
        mujoco.mj_kinematics(model, data)
        for k, S in enumerate(SIDES):
            for i, b in enumerate(hands.bodies[S]):
                for j, n in enumerate(HELD_CANDIDATES):
                    best = reach
                    for g1 in hands.geoms[b]:
                        for g2 in prop_geoms[n]:
                            d0 = np.linalg.norm(data.geom_xpos[g1] - data.geom_xpos[g2]) - model.geom_rbound[g1] - model.geom_rbound[g2]
                            if d0 < best:
                                best = min(best, mujoco.mj_geomDistance(model, data, g1, g2, reach, fromto))
                    gap[t, k, i, j] = best
    loaded = loaded_regions(c)
    held = {}
    for k, S in enumerate(SIDES):
        any_loaded = loaded[S].any(1)
        held[S] = int(np.argmin(np.median(gap[any_loaded, k].min(1), 0)))
    return dict(gap=gap, loaded=loaded, held=held, body_names=hands.body_names)


def audit_state(audit: dict, c: dict, touch: float, near: float) -> np.ndarray:
    """(T, 2, nb) body_state for the kinematic replay."""
    T = len(c["times"])
    names = audit["body_names"]
    nb = len(names["L"])
    state = np.zeros((T, 2, nb), np.int8)
    regions = list(REGIONS)
    for k, S in enumerate(SIDES):
        g = audit["gap"][:, k, :, audit["held"][S]]
        for r, reg in enumerate(regions):
            on = audit["loaded"][S][:, r]
            for short in REGION_BODIES[reg]:
                i = names[S].index(f"{S}_{short}")
                state[on, k, i] = np.maximum(state[on, k, i], np.where(g[on, i] < near, 2, 3))
        state[:, k][g < touch] = 1
    return state


def report_audit(audit: dict, touch: float, near: float) -> None:
    regions = list(REGIONS)
    print("audit (kinematic replay of the mocap): gap of the capsules carrying each loaded glove region to the held prop")
    for k, S in enumerate(SIDES):
        names = audit["body_names"][S]
        g = audit["gap"][:, k, :, audit["held"][S]]
        prop = HELD_CANDIDATES[audit["held"][S]].removeprefix("cleaningtools_")
        print(f"  {SIDE_NAME[S]} hand holds the {prop}")
        for r, reg in enumerate(regions):
            on = audit["loaded"][S][:, r]
            if not on.any():
                continue
            gg = g[on][:, [names.index(f"{S}_{b}") for b in REGION_BODIES[reg]]].min(1)
            print(f"    {reg:24s} loaded {on.mean():4.0%} of frames   gap median {np.median(gg) * 100:4.1f} cm   "
                  f"< {touch * 1000:.0f} mm {np.mean(gg < touch):4.0%}   < {near * 100:.0f} cm {np.mean(gg < near):4.0%}")


def run_physics(c: dict, args: dict, with_props: bool, squeeze: float, ignore: tuple[str, ...], held: dict[str, int] | None) -> tuple[dict, dict]:
    dt = args["dt"]
    model = build(c, with_props, dt, args["masses"], args["props_dir"], args["prop_friction"], args["hand_friction"], args["torsional"])
    data = mujoco.MjData(model)
    ref = Reference(c)
    names = [str(n) for n in c["joint_names"]]
    hands = Hands(model, names)
    T = len(ref.t)
    nb = len(hands.bodies["L"])

    loaded = loaded_regions(c)
    regions = list(REGIONS)
    # Per finger hinge, the frames its finger's glove regions are loaded (flexion hinges only, minus ignored fingers).
    squeeze_on = np.zeros((T, len(names)), bool)
    for jn_i, jn in enumerate(names):
        S, rest = jn.split("_", 1)
        finger = next(f for f in FINGERS if rest.startswith(f))
        if "flex" not in rest or finger in ignore:
            continue
        cols = [r for r, reg in enumerate(regions) if region_finger(reg) == finger]
        squeeze_on[:, jn_i] = loaded[S][:, cols].any(1)

    prop_ids = {n: model.body(n).id for n in HELD_CANDIDATES} if with_props else {}
    prop_qadr = {n: model.jnt_qposadr[model.joint(f"{n}_free").id] for n in prop_ids}
    body_to_prop = {b: n for n, b in prop_ids.items()}
    hand_body = {b for S in SIDES for b in hands.bodies[S]}

    mujoco.mj_resetData(model, data)
    t0 = float(ref.t[0])
    hands.set_pose(data, ref, t0)
    lift = {}
    for n in prop_ids:
        p, q = ref.prop(t0, ref.props.index(n))
        data.qpos[prop_qadr[n] : prop_qadr[n] + 3], data.qpos[prop_qadr[n] + 3 : prop_qadr[n] + 7] = p, q
    mujoco.mj_kinematics(model, data)
    for n, b in prop_ids.items():
        # Start each prop just clear of the table (the trackers put it a few mm off).
        zmin = min((data.geom_xpos[g] + (data.geom_xmat[g].reshape(3, 3) @ model.mesh_vert[model.mesh_vertadr[model.geom_dataid[g]]:
                    model.mesh_vertadr[model.geom_dataid[g]] + model.mesh_vertnum[model.geom_dataid[g]]].T).T)[:, 2].min()
                   for g in range(model.ngeom) if model.geom_bodyid[g] == b)
        lift[n] = TABLE_TOP_HEIGHT + 0.001 - zmin
        data.qpos[prop_qadr[n] + 2] += lift[n]
    mujoco.mj_forward(model, data)

    w = 2 * math.pi * args["wrist_hz"]
    kp, kd = w * w, 2 * w
    M = np.zeros((model.nv, model.nv))
    e, col = np.zeros(model.nv), np.zeros(model.nv)
    sub = round(1.0 / (dt * 60.0))
    rec = {k: [] for k in ("times", "base_pos", "base_quat", "joint_pos", "prop_pos", "prop_quat", "body_state", "wrist_sat")}
    contact_log = []
    steps = int(round((ref.t[-1] - t0) / dt))
    for s in range(steps + 1):
        tq = t0 + s * dt
        fi = ref.frame(tq)
        sat = np.zeros(2, bool)
        for d0 in hands.root_d:
            for i in range(6):
                e[:] = 0.0
                e[d0 + i] = 1.0
                mujoco.mj_mulM(model, data, col, e)
                M[:, d0 + i] = col
        for k in range(2):
            p_r, q_r, v_r, w_r = ref.base(tq, k)
            a, d = hands.root_q[k], hands.root_d[k]
            R = Rotation.from_quat(data.qpos[a + 3 : a + 7], scalar_first=True)
            e_r = R.inv().apply((Rotation.from_quat(q_r, scalar_first=True) * R.inv()).as_rotvec())
            acc = np.concatenate([kp * (p_r - data.qpos[a : a + 3]) + kd * (v_r - data.qvel[d : d + 3]),
                                  kp * e_r + kd * (R.inv().apply(w_r) - data.qvel[d + 3 : d + 6])])
            tau = M[d : d + 6, d : d + 6] @ acc + data.qfrc_bias[d : d + 6]
            f, m = np.linalg.norm(tau[:3]), np.linalg.norm(tau[3:])
            if f > args["wrist_force"]:
                tau[:3] *= args["wrist_force"] / f
                sat[k] = True
            if m > args["wrist_torque"]:
                tau[3:] *= args["wrist_torque"] / m
                sat[k] = True
            data.qfrc_applied[d : d + 6] = tau
        qj, _ = ref.joints(tq)
        data.ctrl[hands.act] = qj + squeeze * squeeze_on[fi]

        touching = np.zeros((2, nb), bool)
        for ci in range(data.ncon):
            con = data.contact[ci]
            b1, b2 = model.geom_bodyid[con.geom1], model.geom_bodyid[con.geom2]
            for hb, pb in ((b1, b2), (b2, b1)):
                if hb in hand_body and pb in body_to_prop:
                    k, i = hands.body_side[hb]
                    if held is None or HELD_CANDIDATES[held[SIDES[k]]] == body_to_prop[pb]:
                        touching[k, i] = True
        if s % sub == 0:
            rec["times"].append(tq)
            rec["base_pos"].append([data.qpos[a : a + 3].copy() for a in hands.root_q])
            rec["base_quat"].append([data.qpos[a + 3 : a + 7].copy() for a in hands.root_q])
            rec["joint_pos"].append(data.qpos[hands.jq].copy())
            pp, pq = [], []
            for j, n in enumerate(ref.props):
                if n in prop_qadr:
                    pp.append(data.qpos[prop_qadr[n] : prop_qadr[n] + 3].copy())
                    pq.append(data.qpos[prop_qadr[n] + 3 : prop_qadr[n] + 7].copy())
                else:
                    p, q = ref.prop(tq, j)
                    pp.append(p)
                    pq.append(q)
            rec["prop_pos"].append(pp)
            rec["prop_quat"].append(pq)
            state = np.zeros((2, nb), np.int8)
            for k, S in enumerate(SIDES):
                for r, reg in enumerate(regions):
                    if loaded[S][fi, r]:
                        for short in REGION_BODIES[reg]:
                            state[k, hands.body_names[S].index(f"{S}_{short}")] = 3
            state[touching] = 1
            rec["body_state"].append(state)
            rec["wrist_sat"].append(sat)
            contact_log.append(touching)
        if s < steps:
            mujoco.mj_step(model, data)

    out = {k: np.array(v) for k, v in rec.items()}
    out["body_names_L"], out["body_names_R"] = np.array(hands.body_names["L"]), np.array(hands.body_names["R"])

    # Metrics against the clip, on the recorded frames.
    tt = out["times"]
    fr = np.array([ref.frame(x) for x in tt])
    pos_err = np.linalg.norm(out["base_pos"] - c["base_pos"][fr], axis=-1)
    rot_err = np.stack([(Rotation.from_quat(out["base_quat"][:, k], scalar_first=True).inv() *
                         Rotation.from_quat(c["base_quat"][fr, k], scalar_first=True)).magnitude() for k in range(2)], 1)
    j_err = np.abs(out["joint_pos"] - c["joint_pos"][fr])
    met = dict(wrist_pos_mm=(np.median(pos_err) * 1000, pos_err.max() * 1000), wrist_rot_deg=(np.degrees(np.median(rot_err)), np.degrees(rot_err.max())),
               joint_deg=(np.degrees(np.median(j_err)), np.degrees(np.percentile(j_err, 95))), wrist_sat=out["wrist_sat"].mean(), lift=lift)
    if with_props:
        props = {}
        for n in prop_ids:
            j = ref.props.index(n)
            e = np.linalg.norm(out["prop_pos"][:, j] - c["prop_pos"][fr, j], axis=-1)
            z0 = c["prop_pos"][0, j, 2]
            lifted = c["prop_pos"][fr, j, 2] > z0 + 0.03
            ang = np.degrees((Rotation.from_quat(out["prop_quat"][:, j], scalar_first=True).inv() *
                              Rotation.from_quat(c["prop_quat"][fr, j], scalar_first=True)).magnitude())
            drop = np.flatnonzero(e > 0.10)
            props[n] = dict(err_cm=(np.median(e) * 100, np.percentile(e, 90) * 100, e.max() * 100), ang_deg=np.median(ang),
                            lifted=lifted.mean(), held_when_lifted=np.mean(e[lifted] < 0.05) if lifted.any() else float("nan"),
                            first_drop=float(tt[drop[0]] - tt[0]) if len(drop) else None)
        met["props"] = props
        touch = np.array(contact_log)
        cons = {}
        for k, S in enumerate(SIDES):
            for r, reg in enumerate(regions):
                on = loaded[S][fr, r]
                if on.any():
                    idx = [hands.body_names[S].index(f"{S}_{b}") for b in REGION_BODIES[reg]]
                    cons[(S, reg)] = (on.mean(), touch[on][:, k, idx].any(1).mean())
        met["region_contact"] = cons
    return out, met


def print_metrics(tag: str, met: dict) -> None:
    print(f"{tag}")
    print(f"  wrist position error median {met['wrist_pos_mm'][0]:.1f} mm (max {met['wrist_pos_mm'][1]:.1f}), "
          f"rotation median {met['wrist_rot_deg'][0]:.2f} deg (max {met['wrist_rot_deg'][1]:.1f}), "
          f"wrench at its limit {met['wrist_sat']:.1%} of frames")
    print(f"  finger hinge error median {met['joint_deg'][0]:.1f} deg, p95 {met['joint_deg'][1]:.1f} deg")
    for n, p in met.get("props", {}).items():
        d = "never" if p["first_drop"] is None else f"{p['first_drop']:.2f} s"
        print(f"  {n.removeprefix('cleaningtools_'):8s} position error median {p['err_cm'][0]:5.1f} cm, p90 {p['err_cm'][1]:5.1f}, "
              f"max {p['err_cm'][2]:5.1f}; angle median {p['ang_deg']:4.0f} deg; lifted {p['lifted']:4.0%} of the clip, "
              f"within 5 cm while lifted {p['held_when_lifted']:4.0%}; first > 10 cm at {d}")
    if "region_contact" in met:
        print("  loaded glove regions touching the held prop in physics:")
        for (S, reg), (frac, hit) in met["region_contact"].items():
            print(f"    {SIDE_NAME[S]:5s} {reg:24s} loaded {frac:4.0%}   touching {hit:4.0%}")


def main(clip: Path = HAND_DATA / "sweep_task023_hands.npz", out_dir: Path = Path("/tmp/capsule_check"),
         props_dir: Path = ROOT / "data/props", squeeze: tuple[float, ...] = (0.0, 0.3, 0.6), ignore: tuple[str, ...] = ("pinky",),
         skip_audit: bool = False, skip_air: bool = False, dt: float = 0.002, wrist_hz: float = 6.0, wrist_force: float = 40.0,
         wrist_torque: float = 4.0, broom_mass: float = 0.10, dustpan_mass: float = 0.20, hand_friction: float = 1.0,
         prop_friction: float = 0.8, torsional: float = 0.005, touch: float = 0.005, near: float = 0.02) -> None:
    """``ignore``: fingers never squeezed; ``wrist_hz``: wrist PD bandwidth; ``wrist_force`` / ``wrist_torque``: wrench limits (N, N m)."""
    c = dict(np.load(clip))
    out_dir.mkdir(parents=True, exist_ok=True)
    args = dict(dt=dt, wrist_hz=wrist_hz, wrist_force=wrist_force, wrist_torque=wrist_torque, props_dir=props_dir,
                masses={"cleaningtools_broom": broom_mass, "cleaningtools_dustpan": dustpan_mass}, hand_friction=hand_friction,
                prop_friction=prop_friction, torsional=torsional)
    held = None
    if not skip_audit:
        audit = kinematic_audit(c, props_dir, touch, near)
        report_audit(audit, touch, near)
        held = audit["held"]
        np.savez_compressed(out_dir / "audit.npz", times=c["times"], base_pos=c["base_pos"], base_quat=c["base_quat"], joint_pos=c["joint_pos"],
                            prop_pos=c["prop_pos"], prop_quat=c["prop_quat"], body_state=audit_state(audit, c, touch, near),
                            body_names_L=np.array(audit["body_names"]["L"]), body_names_R=np.array(audit["body_names"]["R"]),
                            gap=audit["gap"])
        print(f"  wrote {out_dir / 'audit.npz'}")
    if not skip_air:
        out, met = run_physics(c, args, False, 0.0, ignore, None)
        print_metrics("air (no props): hands tracking the mocap in physics", met)
        np.savez_compressed(out_dir / "air.npz", **out)
    for sq in squeeze:
        out, met = run_physics(c, args, True, sq, ignore, held)
        print_metrics(f"hold, squeeze {sq:g} rad (not squeezed: {', '.join(ignore) or 'none'}); props lifted off the table at start by "
                      + ", ".join(f"{n.removeprefix('cleaningtools_')} {v * 1000:+.1f} mm" for n, v in met["lift"].items()), met)
        np.savez_compressed(out_dir / f"hold_sq{sq:g}.npz", **out)


if __name__ == "__main__":
    tyro.cli(main)

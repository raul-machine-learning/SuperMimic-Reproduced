"""Free rigid props (``ROBOT.props``) and the static table of the sweep scene.

Each prop is its own mjlab entity: one body with a free joint, the CoACD convex pieces of
``scripts/assets/build_props.py`` as collision geoms and the visual mesh in geom group 2. Meshes are stored
in meters at human scale and scaled here by the clip's scene scale, like the mocap they were retargeted from.
The mass is set explicitly; the inertia is that of the convex pieces at uniform density, rescaled to it.
The table is a fixed entity, which mjlab wraps in a mocap body positioned per env at reset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import mujoco
import numpy as np
from mjlab.entity import EntityCfg

PROPS_DIR = Path(__file__).resolve().parents[3] / "data/props"
PROP_SOURCES = {"broom": "cleaningtools_broom", "dustpan": "cleaningtools_dustpan"}
PROP_FRICTION = (0.8, 0.005, 0.0001)
PROP_RGBA = {"broom": (0.80, 0.55, 0.30, 1.0), "dustpan": (0.30, 0.50, 0.80, 1.0)}


@dataclass(frozen=True)
class PropSceneCfg:
    scale: float
    table_center: tuple[float, float, float]
    table_half: tuple[float, float, float]
    masses: dict[str, float] = field(default_factory=lambda: {"broom": 0.10, "dustpan": 0.20})
    props_dir: Path = PROPS_DIR

    @staticmethod
    def from_clip(npz: str | Path, **kw) -> PropSceneCfg:
        with np.load(npz) as d:
            assert "prop_pos_w" in d, f"{npz} has no prop reference (scripts/data/add_props_to_clip.py)"
            return PropSceneCfg(float(d["scene_scale"]), tuple(map(float, d["table_center"])), tuple(map(float, d["table_half"])), **kw)


def _prop_spec(name: str, scale: float, props_dir: Path) -> mujoco.MjSpec:
    src = props_dir / PROP_SOURCES[name]
    pieces = sorted(src.glob("collision_*.obj"))
    assert pieces, f"no collision pieces in {src}: run scripts/assets/build_props.py"
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=name)
    body.add_freejoint(name=f"{name}_free")
    for i, f in enumerate(pieces):
        spec.add_mesh(name=f"{name}_c{i:02d}", file=str(f.resolve()), scale=[scale] * 3)
        body.add_geom(name=f"{name}_c{i:02d}_collision", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{name}_c{i:02d}",
                      group=3, condim=3, friction=list(PROP_FRICTION), density=1.0, rgba=[0.5, 0.5, 0.5, 0.0])
    spec.add_mesh(name=f"{name}_visual", file=str((src / "visual.stl").resolve()), scale=[scale] * 3)
    body.add_geom(name=f"{name}_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{name}_visual", contype=0, conaffinity=0,
                  group=2, density=0.0, rgba=list(PROP_RGBA[name]))
    return spec


@cache
def _inertial(name: str, scale: float, props_dir: Path) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    m = _prop_spec(name, scale, props_dir).compile()
    b = m.body(name).id
    return float(m.body_mass[b]), m.body_ipos[b].copy(), m.body_iquat[b].copy(), m.body_inertia[b].copy()


def prop_spec(name: str, scale: float, mass: float, props_dir: Path = PROPS_DIR) -> mujoco.MjSpec:
    m0, ipos, iquat, inertia = _inertial(name, scale, props_dir)
    spec = _prop_spec(name, scale, props_dir)
    body = spec.body(name)
    body.explicitinertial = True
    body.mass, body.ipos, body.iquat, body.inertia = mass, ipos, iquat, inertia * (mass / m0)
    return spec


def table_spec(half: tuple[float, float, float]) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="table")
    body.add_geom(name="table_collision", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half), condim=3,
                  friction=list(PROP_FRICTION), rgba=[0.62, 0.48, 0.34, 1.0])
    return spec


def prop_entity_cfgs(cfg: PropSceneCfg, props: tuple[str, ...]) -> dict[str, EntityCfg]:
    """Scene entities: one per prop (in ``props`` order, after the robot), then the table."""
    out: dict[str, EntityCfg] = {}
    for n in props:
        out[n] = EntityCfg(spec_fn=lambda n=n: prop_spec(n, cfg.scale, cfg.masses[n], cfg.props_dir))
    out["table"] = EntityCfg(spec_fn=lambda: table_spec(cfg.table_half), init_state=EntityCfg.InitialStateCfg(pos=cfg.table_center))
    return out

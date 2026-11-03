"""Build the sweep props (broom, dustpan, trash can) as MuJoCo assets: a visual mesh and a convex decomposition.

  uv run --with coacd --with trimesh python scripts/assets/build_props.py --mocap-dir data/sweep_mocap

Writes ``data/props/<prop>/``: ``visual.stl`` and ``collision_NN.obj`` (CoACD convex pieces,
since MuJoCo collides a mesh as its convex hull and the broom and dustpan are far from convex). Meshes are in
meters in the tracker frame of the recording (the mocap STLs are in centimeters); scale them by the retarget's scene
scale when loading.
"""

from __future__ import annotations

from pathlib import Path

import coacd
import numpy as np
import trimesh
import tyro

ROOT = Path(__file__).resolve().parents[2]
PROPS = {"cleaningtools_broom": "cleaningtools_broom.stl", "cleaningtools_dustpan": "cleaningtools_dustpan.stl",
         "TrashBucket_chip": "trashbucket_chip.stl"}


def main(mocap_dir: Path = ROOT / "data/sweep_mocap", out_dir: Path = ROOT / "data/props",
         threshold: float = 0.03, max_hulls: int = 32, seed: int = 0) -> None:
    """``threshold``: CoACD concavity tolerance (relative to the mesh size); ``max_hulls``: cap on pieces per prop."""
    for name, file in PROPS.items():
        mesh = trimesh.load(mocap_dir / file, force="mesh")
        mesh.apply_scale(0.01)
        d = out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("collision_*.obj"):
            old.unlink()
        mesh.export(d / "visual.stl")
        parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=threshold, max_convex_hull=max_hulls, seed=seed)
        vol = 0.0
        for i, (v, f) in enumerate(parts):
            hull = trimesh.Trimesh(np.asarray(v), np.asarray(f))
            vol += hull.volume
            hull.export(d / f"collision_{i:02d}.obj")
        print(f"{name}: {len(parts)} convex pieces, extents {np.round(mesh.extents * 100, 1)} cm, "
              f"hull volume {vol * 1e6:.0f} cm3 vs mesh {abs(mesh.volume) * 1e6:.0f} cm3")


if __name__ == "__main__":
    tyro.cli(main)

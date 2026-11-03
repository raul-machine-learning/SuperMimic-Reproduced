"""Signed-distance grids of the props (their CoACD convex pieces), in the prop frame, for the glove-pattern reward.

  uv run python scripts/assets/build_prop_sdf.py

Writes ``data/props/<source>/sdf.npz``: ``sdf`` (nx, ny, nz) float16 in m (negative inside, clipped to +-CLIP),
``lo`` (3,) the first voxel center, ``voxel`` the spacing. The surface is sampled every SPACING (outward normals of
the convex pieces); the sign comes from the normal at the nearest sample.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tyro
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from fit_glove_taxels import prop_cloud

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ("cleaningtools_broom", "cleaningtools_dustpan")
CLIP = 0.05


def main(props_dir: Path = ROOT / "data/props", voxel: float = 0.003, margin: float = 0.06, spacing: float = 0.001) -> None:
    for name in SOURCES:
        pts, nrm = prop_cloud(props_dir, name, spacing)
        tree = cKDTree(pts)
        lo, hi = pts.min(0) - margin, pts.max(0) + margin
        n = np.ceil((hi - lo) / voxel).astype(int) + 1
        axes = [lo[i] + voxel * np.arange(n[i]) for i in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        d, i = tree.query(grid, distance_upper_bound=CLIP * 1.5, workers=-1)
        far = ~np.isfinite(d)
        i[far] = 0
        side = np.einsum("ij,ij->i", grid - pts[i], nrm[i])
        sdf = np.where(far, CLIP, np.where(side < 0, -d, d))
        # Deep inside (no sample within the bound) keeps the sign of the nearest inside voxel: none for thin props.
        sdf = np.clip(sdf, -CLIP, CLIP).reshape(*n).astype(np.float16)
        out = props_dir / name / "sdf.npz"
        np.savez_compressed(out, sdf=sdf, lo=lo.astype(np.float32), voxel=np.float32(voxel))
        print(f"wrote {out}: grid {tuple(n)}, extent {np.round(hi - lo, 3)} m, {len(pts)} surface samples, "
              f"inside voxels {(sdf < 0).mean():.2%}")


if __name__ == "__main__":
    tyro.cli(main)

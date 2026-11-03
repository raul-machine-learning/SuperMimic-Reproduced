"""Test data: the sweep mocap clip retargeted to the active robot (``SM_ROBOT``).

Tests that need several clips get the same clip repeated, which still exercises the clip-boundary logic
of the concatenated motion library.
"""

from pathlib import Path

import pytest

from supermimic.robots import ROBOT

ROOT = Path(__file__).resolve().parents[1]
CLIP_NAME = {"g1": "sweep_task023_g1", "g1_inspire": "sweep_task023_g1_inspire", "capsule_hands": "sweep_task023", "capsule_hands_props": "sweep_task023"}[ROBOT.name]
NPZ_DIR = ROOT / {"g1": "data/npz/sweep", "g1_inspire": "data/npz/sweep_inspire", "capsule_hands": "data/npz/capsule_hands", "capsule_hands_props": "data/npz/capsule_hands"}[ROBOT.name]
NPZ = NPZ_DIR / f"{CLIP_NAME}.npz"
G1_CSV = ROOT / "data/csv/sweep/sweep_task023_g1.csv"

requires_sweep = pytest.mark.skipif(not NPZ.exists(), reason=f"{NPZ} not built")


def sweep_files(n: int = 1) -> list[str]:
    return [str(NPZ)] * n

"""Annealed grip assist: grasp offsets on the gripping hands' set-points while the reference glove holds.

Tracking the mocap finger pose exactly barely touches the props, so nothing is carried and the world model never
sees a lifted prop to learn from. The assist adds ``ROBOT.grip_assist`` (joint regex -> rad, a grasp found with
scripts/diag_grasp_search.py) times ``level`` to a hand of ``ROBOT.grip`` on the frames where its glove holds
(switched on ``ROBOT.grip_anticipate`` frames early). The training loop anneals ``level`` from 1 to 0, so the
trained policy supplies the whole grip itself.

The environment's set-point and the imagined set-point both include it, and the replay buffer stores the executed
set-point, so the world model always sees the action that was really applied.
"""

from __future__ import annotations

import re

import torch

from supermimic.robots import ROBOT

_level = 0.0


def set_level(level: float) -> None:
    global _level
    _level = float(level)


def level() -> float:
    return _level


def scheduled_level(iteration: int, hold_iterations: int, decay_iterations: int) -> float:
    """1 up to ``hold_iterations``, then linear to 0 over ``decay_iterations``."""
    if decay_iterations <= 0:
        return 1.0 if iteration < hold_iterations else 0.0
    return min(max(1.0 - (iteration - hold_iterations) / decay_iterations, 0.0), 1.0)


def offsets(joint_names: tuple[str, ...] | list[str], device) -> torch.Tensor:
    """(G, A) full grasp offsets (rad) of each gripping hand: the first matching ``ROBOT.grip_assist`` pattern."""
    out = torch.zeros(len(ROBOT.grip), len(joint_names), device=device)
    for g, (side, _) in enumerate(ROBOT.grip):
        for a, j in enumerate(joint_names):
            if j.startswith(f"{side}_"):
                out[g, a] = next((v for p, v in ROBOT.grip_assist if re.fullmatch(p, j)), 0.0)
    return out


def assist_offset(hold_early: torch.Tensor, full: torch.Tensor) -> torch.Tensor | None:
    """(..., A) set-point offset in rad from the (..., G) anticipated holding flags; None when off."""
    if _level == 0.0 or not ROBOT.grip_assist:
        return None
    return _level * (hold_early.float() @ full)

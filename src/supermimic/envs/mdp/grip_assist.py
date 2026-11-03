"""Annealed grip assist: extra closing of the gripping hands' flex set-points while the reference glove holds.

Tracking the mocap finger pose exactly barely touches the props, so nothing is carried and the world model never
sees a lifted prop to learn from. The assist adds ``level`` rad to every flex joint of a hand of ``ROBOT.grip`` on
the frames where its glove holds (switched on ``ROBOT.grip_anticipate`` frames early). The training loop anneals
``level`` from ``ROBOT.grip_assist`` to 0, so the trained policy supplies the whole grip itself.

The environment's set-point and the imagined set-point both include it, and the replay buffer stores the executed
set-point, so the world model always sees the action that was really applied.
"""

from __future__ import annotations

import torch

from supermimic.robots import ROBOT

_level = 0.0


def set_level(level: float) -> None:
    global _level
    _level = float(level)


def level() -> float:
    return _level


def scheduled_level(iteration: int, hold_iterations: int, decay_iterations: int) -> float:
    """``ROBOT.grip_assist`` up to ``hold_iterations``, then linear to 0 over ``decay_iterations``."""
    if decay_iterations <= 0:
        return ROBOT.grip_assist if iteration < hold_iterations else 0.0
    return ROBOT.grip_assist * min(max(1.0 - (iteration - hold_iterations) / decay_iterations, 0.0), 1.0)


def flex_mask(joint_names: tuple[str, ...] | list[str], device) -> torch.Tensor:
    """(G, A) True for the flex joints of each gripping hand."""
    return torch.tensor([[j.startswith(f"{s}_") and j.endswith("_flex") for j in joint_names] for s, _ in ROBOT.grip],
                        dtype=torch.bool, device=device)


def assist_offset(hold_early: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    """(..., A) set-point offset in rad from the (..., G) anticipated holding flags; None when off."""
    if _level == 0.0 or not ROBOT.grip:
        return None
    return _level * (hold_early.float() @ mask.float())

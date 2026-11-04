"""Contact normal force of each gripping hand's force bodies on the prop it holds (``ROBOT.grip``).

One mjlab contact sensor per hand: primaries are the taxel-carrying bodies (``supermimic.kinematics.glove.body_names``),
the secondary is the hand's prop. Each body keeps its ``SLOTS`` strongest contacts; their normal components are summed.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import torch

from supermimic.robots import ROBOT

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sensor.contact_sensor import ContactSensorCfg

SLOTS = 4


@cache
def _names() -> tuple[tuple[str, ...], ...]:
    from supermimic.kinematics.glove import body_names

    return tuple(tuple(n) for n in body_names())


def num_bodies() -> int:
    return len(_names()[0]) if ROBOT.grip else 0


def sensor_name(side: str) -> str:
    return f"grip_{side}"


def sensor_cfgs() -> tuple[ContactSensorCfg, ...]:
    from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg

    return tuple(
        ContactSensorCfg(
            name=sensor_name(side),
            primary=ContactMatch(mode="body", pattern=tuple(dict.fromkeys(_names()[g])), entity="robot"),
            secondary=ContactMatch(mode="body", pattern=prop, entity=prop),
            fields=("found", "force"),
            reduce="maxforce",
            num_slots=SLOTS,
        )
        for g, (side, prop) in enumerate(ROBOT.grip)
    )


def body_forces(env: ManagerBasedRlEnv) -> torch.Tensor:
    """(N, G, B) normal force (N) of every force body on its hand's prop, 0 in padded body slots."""
    out = torch.zeros(env.num_envs, len(ROBOT.grip), num_bodies(), device=env.device)
    for g, (side, _) in enumerate(ROBOT.grip):
        d = env.scene[sensor_name(side)].data
        fn = torch.where(d.found > 0, d.force[..., 0].clamp(min=0.0), 0.0).view(env.num_envs, -1, SLOTS).sum(-1)
        out[:, g, : fn.shape[1]] = fn
    return out

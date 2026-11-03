"""Grip assist: schedule, flex mask, the library's anticipated holding flags and the goal's holding flags."""

import pytest
import torch

from sweep_data import requires_sweep, sweep_files

from supermimic.envs.mdp import grip_assist
from supermimic.robots import ROBOT

pytestmark = pytest.mark.skipif(not ROBOT.grip, reason="robot has no grip")


def test_schedule_and_mask():
    a0 = ROBOT.grip_assist
    assert grip_assist.scheduled_level(0, 100, 200) == a0
    assert grip_assist.scheduled_level(200, 100, 200) == pytest.approx(0.5 * a0)
    assert grip_assist.scheduled_level(10_000, 100, 200) == 0.0
    m = grip_assist.flex_mask(ROBOT.actuated_joint_names, "cpu")
    for g, (side, _) in enumerate(ROBOT.grip):
        names = [j for j, on in zip(ROBOT.actuated_joint_names, m[g].tolist()) if on]
        assert names and all(j.startswith(f"{side}_") and j.endswith("_flex") and "wrist" not in j for j in names)
    grip_assist.set_level(0.0)
    assert grip_assist.assist_offset(torch.ones(3, len(ROBOT.grip)), m) is None
    grip_assist.set_level(0.5)
    try:
        off = grip_assist.assist_offset(torch.tensor([[1.0] + [0.0] * (len(ROBOT.grip) - 1)]), m)
        assert torch.equal(off[0], 0.5 * m[0].float())
    finally:
        grip_assist.set_level(0.0)


@requires_sweep
def test_library_holding_flags():
    from supermimic.envs.mdp.motion_library import GOAL_FEAT_DIM, MotionLibrary
    from supermimic.envs.tracking_env_cfg import TRACKED_BODIES

    lib = MotionLibrary(sweep_files(2), TRACKED_BODIES, torch.zeros(ROBOT.nj), device="cpu")
    hold, early = lib.grip_hold, lib.grip_hold_early
    assert hold.any() and (early | ~hold).all()  # early covers hold
    n = lib.clips[0].length
    for g in range(len(ROBOT.grip)):
        first = int(hold[:n, g].float().argmax())
        assert early[first - ROBOT.grip_anticipate : first, g].all() and not early[first - ROBOT.grip_anticipate - 1, g]
    one = MotionLibrary(sweep_files(1), TRACKED_BODIES, torch.zeros(ROBOT.nj), device="cpu")
    assert torch.equal(early[:n], one.grip_hold_early) and torch.equal(early[n:], one.grip_hold_early)
    G = len(ROBOT.grip)
    assert torch.equal(lib.goal_feat[:, GOAL_FEAT_DIM - G :], hold.float())

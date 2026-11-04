"""Grip assist: schedule, flex mask, the library's anticipated holding flags and the goal's holding flags."""

import pytest
import torch

from sweep_data import requires_sweep, sweep_files

from supermimic.envs.mdp import grip_assist
from supermimic.robots import ROBOT

pytestmark = pytest.mark.skipif(not ROBOT.grip, reason="robot has no grip")


def test_schedule_and_offsets():
    assert grip_assist.scheduled_level(0, 100, 200) == 1.0
    assert grip_assist.scheduled_level(200, 100, 200) == pytest.approx(0.5)
    assert grip_assist.scheduled_level(10_000, 100, 200) == 0.0
    names = ROBOT.actuated_joint_names
    full = grip_assist.offsets(names, "cpu")
    for g, (side, _) in enumerate(ROBOT.grip):
        moved = [j for j, v in zip(names, full[g].tolist()) if v != 0.0]
        assert len(moved) == 21 and all(j.startswith(f"{side}_") and "wrist" not in j for j in moved), moved
    grip_assist.set_level(0.0)
    assert grip_assist.assist_offset(torch.ones(3, len(ROBOT.grip)), full) is None
    grip_assist.set_level(0.5)
    try:
        off = grip_assist.assist_offset(torch.tensor([[1.0] + [0.0] * (len(ROBOT.grip) - 1)]), full)
        assert torch.equal(off[0], 0.5 * full[0])
    finally:
        grip_assist.set_level(0.0)


def test_tracking_release_weights():
    rel = grip_assist.TrackingRelease(ROBOT.tracked_bodies, ROBOT.hand_joint_names, "cpu")
    h = torch.zeros(2, len(ROBOT.grip))
    h[1, 0] = 1.0  # first gripping hand holds in row 1
    bw, hw = rel.weights(h)
    side = ROBOT.grip[0][0]
    assert (bw[0] == 1).all() and (hw[0] == 1).all()
    for b, w in zip(ROBOT.tracked_bodies, bw[1].tolist()):
        assert w == (0.0 if b.startswith(f"{side}_") and b != f"{side}_hand" else 1.0), b
    for j, w in zip(ROBOT.hand_joint_names, hw[1].tolist()):
        assert w == (0.0 if j.startswith(f"{side}_") else 1.0), j


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
    # Start frames: held frames map back to the last free frame of their clip, free frames to themselves.
    fs = lib.free_start
    busy = early.any(-1)
    assert torch.equal(fs[~busy], torch.arange(lib.num_frames)[~busy])
    assert not busy[fs].any() and (lib.frame_clip[fs] == lib.frame_clip).all()
    first = int(busy[:n].float().argmax())
    assert (fs[first : first + 50] == first - 1).all()

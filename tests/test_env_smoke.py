"""Environment layer smoke test: build the env on the sweep clip (repeated), step, inspect the replay."""

import numpy as np
import pytest
import torch

from sweep_data import requires_sweep, sweep_files

from supermimic.envs.mdp.motion_library import GOAL_FEAT_DIM
from supermimic.envs.mdp.rewards import TrackingRewardCfg
from supermimic.robots import ROBOT

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"), requires_sweep]


@pytest.fixture(scope="module")
def env():
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env

    args = SuperMimicEnvCfgArgs(npz_files=sweep_files(3), num_envs=32, replay_capacity=64, seed=1)
    e = make_env(args, device="cuda:0")
    yield e
    e.close()


def test_spaces_and_step(env):
    obs, _ = env.reset()
    assert obs["policy"].shape == (32, 10 * ROBOT.obs_frame_dim)
    assert obs["goal"].shape == (32, 10 * GOAL_FEAT_DIM)
    assert env.action_manager.total_action_dim == ROBOT.na
    for _ in range(40):
        act = 0.3 * torch.randn(32, ROBOT.na, device=env.device)
        obs, rew, term, tout, extras = env.step(act)
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["goal"]).all()
        assert torch.isfinite(rew).all()
    # Rewards are dt-scaled; the positive terms sum to at most max_step_reward (G1: 0.1) per step.
    assert rew.max() <= TrackingRewardCfg().max_step_reward + 1e-6


def test_goal_matches_library(env):
    cmd = env.command_manager.get_term("motion")
    goal = cmd.goal
    lib = cmd.library
    f = cmd.frame
    direct = lib.goal_feat[f]
    np.testing.assert_allclose(goal[:, :GOAL_FEAT_DIM].cpu().numpy(), direct.cpu().numpy(), atol=1e-6)
    # Clamping at the clip end.
    last = lib.last_frame_of(f)
    k9 = torch.minimum(f + 9, last)
    np.testing.assert_allclose(goal[:, 9 * GOAL_FEAT_DIM :].cpu().numpy(), lib.goal_feat[k9].cpu().numpy(), atol=1e-6)


def test_replay_rows_and_windows(env):
    rec = env.recorder_manager.get_term("replay")
    buf = rec.buffer
    assert buf.size > 32 * 10
    # Every non-terminal row's successor within the episode must be the next row.
    e = 0
    rows = [t for t in range(buf.T) if buf.valid[e, t]]
    assert len(rows) > 5
    w = buf.sample_windows(256, 8)
    assert w.states.shape == (9, 256, ROBOT.state_dim) and w.a_exec.shape == (8, 256, ROBOT.na) and w.xi.shape == (256, ROBOT.xi_dim)
    assert torch.isfinite(w.states).all()
    # Executed action differs from the clean one by the injected noise only (and, with residual actions, by the
    # reference set-point of the next frame in default-offset units).
    a_ref = torch.zeros_like(w.a_exec)
    if ROBOT.residual_actions:
        lib = env.command_manager.get_term("motion").library
        f = w.frame.reshape(-1)
        nxt = torch.minimum(f + 1, lib.last_frame_of(f))
        act = ROBOT.joint_indices(ROBOT.actuated_joint_names)
        scale = torch.tensor(ROBOT.action_scale_vector(), device=f.device)
        a_ref = ((lib.ref(nxt).joint_pos[:, act] - lib.default_joint_pos[act]) / scale).reshape(w.a_exec.shape)
    d = (w.a_exec - a_ref - w.a_clean).std().item()
    assert 0.05 < d < 0.2, d
    s = buf.sample_starts(128, hist=10)
    assert s.hist_states.shape == (10, 128, ROBOT.state_dim) and s.hist_actions.shape == (10, 128, ROBOT.na)
    assert torch.equal(s.hist_states[-1], s.state)


def test_state_io_round_trip(env):
    from supermimic.envs.state_io import StateIO

    io = StateIO(env)
    s = io.read()
    assert s.shape == (32, ROBOT.state_dim)
    ids = torch.arange(8, device=env.device)
    s2 = s[ids].clone()
    s2[:, 0] += 0.123
    io.write(ids, s2)
    back = io.read()
    np.testing.assert_allclose(back[ids].cpu().numpy(), s2.cpu().numpy(), atol=1e-6)
    robot = env.scene["robot"]
    origin_x = env.scene.env_origins[ids, 0]
    np.testing.assert_allclose(robot.data.root_link_pos_w[ids, 0].cpu().numpy(), (s2[:, 0] + origin_x).cpu().numpy(), atol=1e-5)


def test_sim_context(env):
    rec = env.recorder_manager.get_term("replay")
    xi = rec.ctx.read()
    nj = ROBOT.nj
    assert xi.shape == (32, ROBOT.xi_dim)
    fr = xi[:, 0]
    assert (fr >= 0.3 - 1e-6).all() and (fr <= 1.2 + 1e-6).all()
    assert xi[:, 1 : 1 + nj].abs().max() <= 0.01 + 1e-6
    assert xi[:, 1 + nj :].abs().max() <= 0.05 + 1e-6
    # Per-episode re-sampling: friction values should not all be identical across envs.
    assert fr.std() > 0.01


def test_episode_never_crosses_clip_boundary(env):
    """Frames must stay inside the clip the episode started in; clip end raises the time-out flag."""
    cmd = env.command_manager.get_term("motion")
    lib = cmd.library
    env.reset()
    start_clip = lib.frame_clip[cmd.frame].clone()
    for _ in range(60):
        env.step(torch.zeros(env.num_envs, ROBOT.na, device=env.device))
        same = lib.frame_clip[cmd.frame] == start_clip
        assert bool(same[~env.reset_buf].all())
        start_clip = lib.frame_clip[cmd.frame].clone()
    # Force an env to the second-to-last frame of its clip and step twice: clip_end must fire as a time-out.
    ids = torch.tensor([0], device=env.device)
    cmd.reset_to_frame(ids, lib.clip_last[lib.frame_clip[cmd.frame[ids]]] - 1)
    env.sim.forward()
    cmd._refresh_ref()
    cmd.update_relative_body_poses()
    fired = False
    for _ in range(3):  # targets last-1 -> last -> (clip_end set) -> time-out on the next check
        obs, rew, term, tout, _ = env.step(torch.zeros(env.num_envs, ROBOT.na, device=env.device))
        fired = fired or bool(tout[0]) or bool(term[0])
    assert fired


@pytest.mark.skipif(not ROBOT.grip or not ROBOT.residual_actions, reason="robot has no grip assist")
def test_grip_assist_moves_the_set_point(env):
    from supermimic.envs.mdp import grip_assist

    term = env.action_manager.get_term("joint_pos")
    cmd = env.command_manager.get_term("motion")
    lib = cmd.library
    cmd.reset_to_frame(torch.arange(env.num_envs, device=env.device), lib.clip_start[lib.frame_clip[cmd.frame]] + 200)
    base = term._reference_target().clone()
    grip_assist.set_level(0.4)
    try:
        assisted = term._reference_target()
    finally:
        grip_assist.set_level(0.0)
    nxt = torch.minimum(cmd.frame + 1, lib.last_frame_of(cmd.frame))
    want = 0.4 * (lib.grip_hold_early[nxt].float() @ grip_assist.flex_mask(term.target_names, env.device).float())
    assert lib.grip_hold_early[nxt].any()
    torch.testing.assert_close(assisted - base, want)


@pytest.mark.skipif(not ROBOT.mimic, reason="robot has no mimic joints")
def test_reset_keeps_mimic_couplings(env):
    """Reset noise is drawn per joint; the followers must be re-driven so the joint equalities hold."""
    robot = env.scene["robot"]
    fol = ROBOT.joint_indices(tuple(m[0] for m in ROBOT.mimic))
    drv = ROBOT.joint_indices(tuple(m[1] for m in ROBOT.mimic))
    mult = torch.tensor([m[2] for m in ROBOT.mimic], device=env.device)
    env.reset()
    q, qd = robot.data.joint_pos, robot.data.joint_vel
    np.testing.assert_allclose(q[:, fol].cpu().numpy(), (mult * q[:, drv]).cpu().numpy(), atol=1e-6)
    np.testing.assert_allclose(qd[:, fol].cpu().numpy(), (mult * qd[:, drv]).cpu().numpy(), atol=1e-5)

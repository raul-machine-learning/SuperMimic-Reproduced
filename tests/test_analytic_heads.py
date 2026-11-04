"""Analytic reward / observation / termination must match mjlab's env computations (plan 2.8, 2.9)."""

import numpy as np
import pytest
import torch

from sweep_data import requires_sweep, sweep_files

from supermimic.robots import ROBOT

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"), requires_sweep]
NJ, NA = ROBOT.nj, ROBOT.na
TRACKED = ROBOT.tracked_bodies


@pytest.fixture(scope="module")
def setup():
    torch.backends.cuda.matmul.allow_tf32 = False
    from mjlab.tasks.tracking import mdp as track_mdp

    from supermimic.data.motion_convert import load_robot_model
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.state_io import StateIO
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.world_model.heads import AnalyticHeads

    args = SuperMimicEnvCfgArgs(npz_files=sweep_files(2), num_envs=16, replay_capacity=64, obs_noise=False, seed=3)
    env = make_env(args, device="cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False  # make_env turns TF32 on; keep the comparison exact
    robot = env.scene["robot"]
    fk = G1Kinematics(load_robot_model(), device=env.device, dtype=torch.float32)
    heads = AnalyticHeads(
        fk,
        TRACKED_BODIES,
        ANCHOR_BODY,
        ROOT_BODY,
        soft_limits=robot.data.soft_joint_pos_limits[0].clone(),
        default_joint_pos=robot.data.default_joint_pos[0].clone(),
    )
    yield dict(env=env, heads=heads, io=StateIO(env), track_mdp=track_mdp, robot=robot)
    env.close()


def _ref_from_command(cmd):
    from supermimic.world_model.heads import ReferenceFrame

    lib, f = cmd.library, cmd.frame
    r = lib.ref(f)
    return ReferenceFrame(
        body_pos_w=lib.body_pos_w[f], body_quat_w=lib.body_quat_w[f],
        body_lin_vel_w=lib.body_lin_vel_w[f], body_ang_vel_w=lib.body_ang_vel_w[f],
        root_pos_w=lib.root_pos_w[f], root_quat_w=lib.root_quat_w[f], joint_pos=lib.joint_pos[f],
        prop_pos_w=r.prop_pos_w, prop_quat_w=r.prop_quat_w,
        taxel_force=lib.taxel_force[f] if ROBOT.grip else None,
        grip_hold_early=lib.grip_hold_early[f].float() if ROBOT.grip else None,
    )


def test_reward_terms_match_mjlab(setup):
    env, heads, io, mdp, robot = setup["env"], setup["heads"], setup["io"], setup["track_mdp"], setup["robot"]
    from mjlab.envs import mdp as base_mdp
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    from supermimic.envs.mdp import rewards as sm_rew
    from supermimic.envs.mdp.rewards import motion_hand_joint_position_error_exp

    env.reset()
    cmd = env.command_manager.get_term("motion")
    if ROBOT.grip:
        # Start inside the holding windows (with some envs before them), where the finger tracking is released.
        lib = cmd.library
        ids = torch.arange(env.num_envs, device=env.device)
        cmd.reset_to_frame(ids, lib.clip_start[lib.frame_clip[cmd.frame]] + 100 + 8 * ids)
        env.sim.forward()
        cmd._refresh_ref()
        cmd.update_relative_body_poses()
        assert lib.grip_hold_early[cmd.frame].any(-1).float().mean() > 0.5
    worst = {}
    for step in range(25):
        act = 0.5 * torch.randn(env.num_envs, NA, device=env.device)
        env.step(act)
        # Post-step: sim.forward() has run and the command advanced; evaluate both on this instant.
        s = io.read()
        grip = None
        if ROBOT.grip:
            from supermimic.envs.mdp.grip_force import body_forces

            grip = (heads.grip_distances(s, heads.robot_bodies(s)), body_forces(env))
        ours = heads.reward(s, env.action_manager.action, env.action_manager.prev_action, _ref_from_command(cmd), grip=grip)
        asset_cfg = SceneEntityCfg("robot", joint_names=ROBOT.limit_joint_names)
        asset_cfg.resolve(env.scene)
        rc = heads.reward_cfg
        theirs = {
            "motion_global_root_pos": 0.5 * mdp.motion_global_anchor_position_error_exp(env, "motion", rc.std_root_pos),
            "motion_global_root_ori": 0.5 * mdp.motion_global_anchor_orientation_error_exp(env, "motion", rc.std_root_ori),
            "motion_body_pos": mdp.motion_relative_body_position_error_exp(env, "motion", rc.std_body_pos),
            "motion_body_ori": mdp.motion_relative_body_orientation_error_exp(env, "motion", rc.std_body_ori),
            "motion_body_lin_vel": mdp.motion_global_body_linear_velocity_error_exp(env, "motion", rc.std_body_lin_vel),
            "motion_body_ang_vel": mdp.motion_global_body_angular_velocity_error_exp(env, "motion", rc.std_body_ang_vel),
            "action_rate_l2": -0.1 * base_mdp.action_rate_l2(env),
            "joint_limit": -2.0 * base_mdp.joint_pos_limits(env, asset_cfg),
        }
        if ROBOT.grip:
            theirs["motion_body_pos"] = sm_rew.motion_body_position_released_exp(env, "motion", rc.std_body_pos)
            theirs["motion_body_ori"] = sm_rew.motion_body_orientation_released_exp(env, "motion", rc.std_body_ori)
            theirs["motion_body_lin_vel"] = sm_rew.motion_body_lin_vel_released_exp(env, "motion", rc.std_body_lin_vel)
            theirs["motion_body_ang_vel"] = sm_rew.motion_body_ang_vel_released_exp(env, "motion", rc.std_body_ang_vel)
        if ROBOT.hand_joint_names:
            hand_ids = ROBOT.joint_indices(ROBOT.hand_joint_names)
            theirs["motion_hand_joint_pos"] = motion_hand_joint_position_error_exp(env, "motion", 0.3, hand_ids)
        if ROBOT.props:
            theirs["motion_prop_pos"] = rc.w_prop_pos * sm_rew.motion_prop_position_error_exp(env, "motion", rc.std_prop_pos)
            theirs["motion_prop_ori"] = rc.w_prop_ori * sm_rew.motion_prop_orientation_error_exp(env, "motion", rc.std_prop_ori)
        if ROBOT.grip:
            theirs["grip_force"] = rc.w_grip * sm_rew.grip_force_agreement(env, "motion")
        assert set(theirs) == set(ours) - {"total"}
        for k, v in theirs.items():
            d = (ours[k] / 0.02 - v).abs().max().item()
            worst[k] = max(worst.get(k, 0.0), d)
    for k, d in worst.items():
        assert d < 2e-5, (k, d, worst)


def test_observation_head_matches_env(setup):
    env, heads, io, robot = setup["env"], setup["heads"], setup["io"], setup["robot"]
    from supermimic.world_model.heads import OBS_TERMS, analytic_observation, flatten_history_term_major

    cmd = env.command_manager.get_term("motion")
    cmd.forced_frame[:] = cmd.library.clip_start[0]  # far from the clip end: no time-out within the rollout
    obs, _ = env.reset()
    cmd.forced_frame[:] = -1
    rec = env.recorder_manager.get_term("replay")
    q0 = robot.data.default_joint_pos[0]
    hist = []
    for step in range(14):
        act = 0.2 * torch.randn(env.num_envs, NA, device=env.device)
        obs, *_ = env.step(act)
        if env.reset_buf.any():
            pytest.skip("an env reset during the short rollout; rerun")  # keeps the test simple and exact
        s = io.read()
        o = analytic_observation(s, env.action_manager.action, rec.ctx.read(), q0)
        hist.append(o)
        newest = obs["policy"].view(env.num_envs, -1)
        # Newest frame of each term in the term-major layout.
        start = 0
        for name, d in OBS_TERMS:
            block = newest[:, start * 10 : (start + d) * 10].reshape(env.num_envs, 10, d)  # time-major inside a term
            env_newest = block[:, -1, :]
            our = o[:, start : start + d]
            np.testing.assert_allclose(our.cpu().numpy(), env_newest.cpu().numpy(), atol=2e-5, err_msg=name)
            start += d
    full = flatten_history_term_major(torch.stack(hist[-10:], 0))
    np.testing.assert_allclose(full.cpu().numpy(), obs["policy"].cpu().numpy(), atol=2e-5)


def test_termination_matches_env(setup):
    env, heads, io = setup["env"], setup["heads"], setup["io"]
    from supermimic.envs.mdp import terminations as sm_term

    env.reset()
    cmd = env.command_manager.get_term("motion")
    for _ in range(10):
        env.step(1.0 * torch.randn(env.num_envs, NA, device=env.device))
        s = io.read()
        ref = _ref_from_command(cmd)
        f = heads.termination_features(s, ref)
        th, tt = ROBOT.termination_thresholds
        if ROBOT.termination == "wrists":
            from mjlab.utils.lab_api.math import quat_error_magnitude

            theirs_h = sm_term.bad_wrist_pos(env, "motion", th)
            theirs_t = sm_term.bad_wrist_ori(env, "motion", tt)
            # Independent check: the hand bodies' world poses (FK of the state) against the reference bodies.
            b = heads.robot_bodies(s)
            hands = [TRACKED.index(f"{S}_hand") for S in "LR"]
            dp = (b["pos"][:, hands] - ref.body_pos_w[:, hands]).norm(dim=-1).amax(-1)
            da = quat_error_magnitude(b["quat"][:, hands], ref.body_quat_w[:, hands]).amax(-1)
            if ROBOT.props:
                prop = (cmd.sim_prop_pos_w - cmd.prop_pos_w).norm(dim=-1).amax(-1)
                dp = torch.maximum(dp, prop * (th / ROBOT.prop_drop_threshold))
                theirs_h = theirs_h | sm_term.bad_prop_pos(env, "motion", ROBOT.prop_drop_threshold)
            np.testing.assert_allclose(f[:, 0].cpu().numpy(), dp.cpu().numpy(), atol=2e-5)
            np.testing.assert_allclose(f[:, 1].cpu().numpy(), da.cpu().numpy(), atol=1e-3)
        else:
            theirs_h = sm_term.bad_root_height(env, "motion", th)
            theirs_t = sm_term.bad_root_tilt(env, "motion", tt)
            # Compare the continuous quantities (booleans could flip on ties).
            h_env = (cmd.root_pos_w[:, 2] - cmd.robot_root_pos_w[:, 2])
            np.testing.assert_allclose(f[:, 0].cpu().numpy(), h_env.cpu().numpy(), atol=2e-5)
        assert torch.equal((f[:, 0].abs() > th), theirs_h) or (f[:, 0].abs() - th).abs().min() < 1e-4
        assert torch.equal((f[:, 1].abs() > tt), theirs_t) or (f[:, 1].abs() - tt).abs().min() < 1e-4


def test_anchor_relative_obs_matches_env():
    """The analytic anchor-relative term (from state via FK) equals mjlab's observation terms."""
    from supermimic.data.motion_convert import load_robot_model
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import GOAL_FEAT_DIM
    from supermimic.envs.state_io import StateIO
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.world_model.heads import AnalyticHeads, anchor_relative_obs

    env = make_env(SuperMimicEnvCfgArgs(npz_files=sweep_files(), num_envs=8, obs_noise=False, anchor_obs=True, record_transitions=False, seed=4), "cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        robot = env.scene["robot"]
        heads = AnalyticHeads(G1Kinematics(load_robot_model(), device=env.device), TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone())
        io = StateIO(env)
        obs, _ = env.reset()
        assert obs["goal"].shape == (8, 10 * GOAL_FEAT_DIM + 9)
        cmd = env.command_manager.get_term("motion")
        for _ in range(5):
            obs, *_ = env.step(0.3 * torch.randn(8, NA, device=env.device))
            s = io.read()
            b = heads.robot_bodies(s)
            ai = heads.anchor_index
            ref = _ref_from_command(cmd)
            ours = anchor_relative_obs(b["pos"][:, ai], b["quat"][:, ai], ref.body_pos_w[:, ai], ref.body_quat_w[:, ai])
            np.testing.assert_allclose(ours.cpu().numpy(), obs["goal"][:, 10 * GOAL_FEAT_DIM :].cpu().numpy(), atol=3e-5)
    finally:
        env.close()

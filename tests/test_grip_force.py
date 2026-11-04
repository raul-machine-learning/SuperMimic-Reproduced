"""Force grip: glove force reference, taxel force spreading and agreement, contact sensors, replay labels, force head."""

import numpy as np
import pytest
import torch

from sweep_data import requires_sweep, sweep_files

from supermimic.robots import ROBOT

pytestmark = pytest.mark.skipif(not ROBOT.grip, reason="robot has no grip")


@pytest.fixture(scope="module")
def glove():
    from supermimic.kinematics.glove import GloveModel

    return GloveModel(lambda names: torch.arange(len(names)), "cpu")


def test_taxel_forces_conserve_body_force(glove):
    from supermimic.kinematics.glove import NTAXEL

    G, B = len(ROBOT.grip), glove.num_bodies
    torch.manual_seed(0)
    d = 0.01 * torch.randn(5, G, NTAXEL)
    bf = 10.0 * torch.rand(5, G, B)
    f = glove.taxel_forces(d, bf)
    per_body = torch.einsum("nht,htb->nhb", f, glove.onehot)
    used = glove.onehot.sum(1) > 0  # (G, B) bodies that carry taxels
    assert torch.allclose(per_body[:, used], bf[:, used], rtol=1e-5)
    # Taxels nearer the prop take a larger share of their body's force.
    near = d.clone()
    near[:, 0, 0] = -0.002
    far = d.clone()
    far[:, 0, 0] = 0.02
    assert (glove.taxel_forces(near, bf)[:, 0, 0] > glove.taxel_forces(far, bf)[:, 0, 0]).all()


def test_force_agreement(glove):
    from supermimic.kinematics.glove import NTAXEL

    G = len(ROBOT.grip)
    ref = torch.zeros(1, G, NTAXEL)
    ref[..., :10] = 1.0
    assert torch.allclose(glove.force_agreement(ref, ref), torch.ones(1, G))
    assert torch.allclose(glove.force_agreement(torch.zeros_like(ref), torch.zeros_like(ref)), torch.ones(1, G))  # free hand
    scores = [glove.force_agreement(k * ref, ref)[0, 0].item() for k in (0.0, 0.5, 1.0, 2.0, 4.0)]
    assert scores[0] < scores[1] < scores[2] and scores[2] > scores[3] > scores[4]  # too soft and too hard both lose
    moved = torch.roll(ref, 20, -1)  # same total force on other taxels
    assert glove.force_agreement(moved, ref)[0, 0] < 0.1


@requires_sweep
def test_library_force_reference():
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.envs.tracking_env_cfg import TRACKED_BODIES
    from supermimic.kinematics.glove import NEWTONS_PER_UNIT

    lib = MotionLibrary(sweep_files(1), TRACKED_BODIES, torch.zeros(ROBOT.nj), device="cpu")
    f = lib.taxel_force
    assert f.shape == lib.taxel_pressed.shape and (f >= 0).all()
    assert (f.sum(-1)[~lib.grip_hold] == 0).all()  # no reference force while a hand is free
    with np.load(sweep_files(1)[0]) as d:
        sides = ["LR".index(s) for s, _ in ROBOT.grip]
        np.testing.assert_allclose(f.numpy(), d["taxel_load"][:, sides] * NEWTONS_PER_UNIT, rtol=1e-6)
    total = f.sum(-1)[lib.grip_hold]
    assert 5.0 < float(total.median()) < 40.0  # newtons: a grasp of a broom or a dustpan


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@requires_sweep
def test_sensors_and_replay_labels():
    """Replaying the reference with the full grasp assist presses the holding hands on their props; the replay
    buffer stores the sensors' forces next to each state."""
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp import grip_assist
    from supermimic.envs.mdp.grip_force import body_forces

    env = make_env(SuperMimicEnvCfgArgs(npz_files=sweep_files(1), num_envs=4, replay_capacity=400, obs_noise=False, reset_noise=False,
                                        domain_randomization=False, action_noise_std=0.0, sampling_mode="start", seed=0), "cuda:0")
    grip_assist.set_level(1.0)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        cmd.forced_frame[:] = lib.clip_start[0]
        env.reset()
        rec = env.recorder_manager.get_term("replay")
        hold_steps = torch.zeros(len(ROBOT.grip), device=env.device)
        pressed = torch.zeros(len(ROBOT.grip), device=env.device)
        for _ in range(300):
            env.step(torch.zeros(env.num_envs, ROBOT.na, device=env.device))
            if env.reset_buf.any():
                break
            f = body_forces(env)
            h = lib.grip_hold[cmd.frame].float()  # (n, G)
            hold_steps += h.sum(0)
            pressed += ((f.sum(-1) > 1.0).float() * h).sum(0)
            np.testing.assert_allclose(rec.prev_force.cpu().numpy(), f.flatten(1).cpu().numpy())
        assert (hold_steps > 0).all()
        assert (pressed / hold_steps > 0.8).all(), pressed / hold_steps
        w = rec.buffer.sample_windows(64, 8, allow_short=True)
        assert w.forces.shape == (9, 64, len(ROBOT.grip) * body_forces(env).shape[-1]) and torch.isfinite(w.forces).all()
        assert (w.forces >= 0).all() and w.forces.sum() > 0
    finally:
        grip_assist.set_level(0.0)
        env.close()


@requires_sweep
def test_force_head_trains_without_moving_dynamics():
    from supermimic.data.motion_convert import load_robot_model
    from supermimic.envs.mdp.grip_force import num_bodies
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.world_model.heads import AnalyticHeads
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    lib = MotionLibrary(sweep_files(1), TRACKED_BODIES, torch.zeros(ROBOT.nj), device="cpu")
    fk = G1Kinematics(load_robot_model(), device="cpu")
    heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, torch.zeros(ROBOT.nj, 2), torch.zeros(ROBOT.nj))
    wm = WorldModel(WorldModelCfg(hidden=(64,), force_hidden=(32,)), heads, "cpu")
    torch.nn.init.normal_(wm.dynamics.mlp.net[-1].weight, std=1e-3)
    frames = torch.tensor([150, 300, 450])
    n = len(frames)
    s = lib._ref[frames].float()  # reference states (the library stores them in the state layout)
    r = lib.ref(frames)
    a = torch.zeros(n, ROBOT.na)
    xi = torch.zeros(n, ROBOT.xi_dim)
    f_true = torch.rand(n, len(ROBOT.grip) * num_bodies())
    _, sq, bce, _, fl, f_err = wm._loss_step(s, a, xi, s.clone(), torch.zeros(n), r.root_pos_w, r.root_quat_w, r.joint_pos, r.prop_pos_w, f_true)
    assert fl.shape == (n,) and f_err.shape == (n,) and torch.isfinite(fl).all()
    fl.sum().backward()
    assert all(p.grad is None or not p.grad.any() for p in wm.dynamics.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in wm.force_head.parameters())

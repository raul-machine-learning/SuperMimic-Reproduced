"""World-model unit tests: state spec consistency, identity at init, gradients, and a quick overfit."""

import numpy as np
import pytest
import torch

from sweep_data import NPZ, sweep_files

from supermimic.robots import ROBOT

NJ, NA, SD, DD = ROBOT.nj, ROBOT.na, ROBOT.state_dim, ROBOT.delta_dim
_, _, JP, LV, _, JV = (slice(a, b) for a, b in ((0, 3), (3, 7), (7, 7 + NJ), (7 + NJ, 10 + NJ), (10 + NJ, 13 + NJ), (13 + NJ, 13 + 2 * NJ)))


def _random_states(n, device, dtype=torch.float32):
    s = torch.randn(n, SD, device=device, dtype=dtype)
    s[:, 3:7] = s[:, 3:7] / s[:, 3:7].norm(dim=-1, keepdim=True)
    for k in range(ROBOT.num_props):
        q = slice(ROBOT.robot_state_dim + 13 * k + 3, ROBOT.robot_state_dim + 13 * k + 7)
        s[:, q] = s[:, q] / s[:, q].norm(dim=-1, keepdim=True)
    return s


def test_state_spec_identity_and_errors():
    from supermimic.world_model.state_spec import StateSpec, StateSpecCfg

    dev = "cpu"
    spec = StateSpec(StateSpecCfg(explicit_integration=False), dev, torch.float64)
    s = _random_states(64, dev, torch.float64)
    s2 = spec.integrate(s, torch.zeros(64, DD, dtype=torch.float64))
    np.testing.assert_allclose(s2.numpy(), s.numpy(), atol=1e-12)
    assert spec.normalized_sq_error(s, s).max() < 1e-20
    # A pure yaw rotation + xy translation of the state leaves the canonical features unchanged.
    from supermimic.kinematics.rotations import quat_apply, quat_from_yaw, quat_mul

    yaw = torch.rand(64, dtype=torch.float64) * 6.28 - 3.14
    qy = quat_from_yaw(yaw)
    s_rot = s.clone()
    s_rot[:, 0:2] += torch.randn(64, 2, dtype=torch.float64)
    s_rot[:, 3:7] = quat_mul(qy, s[:, 3:7])
    s_rot[:, LV] = quat_apply(qy, s[:, LV])
    for k in range(ROBOT.num_props):
        b = ROBOT.robot_state_dim + 13 * k
        s_rot[:, b : b + 3] = s_rot[:, 0:3] + quat_apply(qy, s[:, b : b + 3] - s[:, 0:3])
        s_rot[:, b + 3 : b + 7] = quat_mul(qy, s[:, b + 3 : b + 7])
        s_rot[:, b + 7 : b + 10] = quat_apply(qy, s[:, b + 7 : b + 10])
    np.testing.assert_allclose(spec.canonical(s_rot).numpy(), spec.canonical(s).numpy(), atol=1e-9)
    # Explicit integration with zero delta = constant-velocity extrapolation.
    spec2 = StateSpec(StateSpecCfg(explicit_integration=True), dev, torch.float64)
    s3 = spec2.integrate(s, torch.zeros(64, DD, dtype=torch.float64))
    np.testing.assert_allclose(s3[:, 0:3].numpy(), (s[:, 0:3] + 0.02 * s[:, LV]).numpy(), atol=1e-12)
    np.testing.assert_allclose(s3[:, JP].numpy(), (s[:, JP] + 0.02 * s[:, JV]).numpy(), atol=1e-12)


def test_short_windows_reach_quick_failures():
    """Episodes shorter than the window are sampled with ``allow_short``, padded with their terminal row."""
    from supermimic.data.replay_buffer import ReplayBuffer

    b = ReplayBuffer(1, 40, 1, 1, 1, "cpu")
    row, one = 0, torch.tensor([0])
    for L in (2, 12, 1, 5):  # transitions per episode; each ends with a terminal row
        for i in range(L + 1):
            term = i == L
            b.add(one, torch.tensor([[float(row)]]), torch.ones(1, 1), torch.ones(1, 1), torch.zeros(1, 1), torch.zeros(1),
                  torch.tensor([float(term)]), torch.tensor([i]), one, torch.tensor([term]), torch.tensor([i == 0]))
            row += 1
        b.end_episode(one)
    k = 4
    assert b.window_mask(k).nonzero()[:, 1].tolist() == [3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19]
    short = b.window_mask(k, True).nonzero()[:, 1].tolist()
    assert {0, 1, 12, 13, 14, 16} <= set(short) and not {2, 15, 17} & set(short)
    w = b.sample_windows(400, k, True)
    starts = w.states[0, :, 0].long()
    i0 = (starts == 0).nonzero()[0, 0]
    assert w.states[:, i0, 0].tolist() == [0.0, 1.0, 2.0, 2.0, 2.0]
    assert w.mask[:, i0].tolist() == [1.0, 1.0, 0.0, 0.0] and w.a_exec[:, i0, 0].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert w.done[1, i0] == 1.0
    full = starts == 5
    assert (w.mask[:, full] == 1).all()


@pytest.mark.skipif(not torch.cuda.is_available() or not NPZ.exists(), reason="needs CUDA + the sweep npz")
def test_world_model_overfits_replay_windows():
    """Collect a little data with a random policy, fit the WM for a few hundred steps: error must drop a lot."""
    from supermimic.data.motion_convert import load_robot_model
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.world_model.heads import AnalyticHeads
    from supermimic.world_model.wm_trainer import WMTrainerCfg, WorldModelTrainer
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    env = make_env(SuperMimicEnvCfgArgs(npz_files=sweep_files(2), num_envs=256, replay_capacity=400, seed=5), "cuda:0")
    try:
        env.reset()
        for _ in range(300):
            env.step(0.3 * torch.randn(256, NA, device=env.device))
        rec = env.recorder_manager.get_term("replay")
        robot = env.scene["robot"]
        fk = G1Kinematics(load_robot_model(), device=env.device)
        heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone())
        wm = WorldModel(WorldModelCfg(hidden=(512, 512, 512)), heads, env.device)
        lib = env.command_manager.get_term("motion").library
        tr = WorldModelTrainer(wm, lib, WMTrainerCfg(batch_size=512, batches_per_update=1, lr=1e-3))
        e0 = tr.online_error(rec.buffer, n=1024, k=8)
        for _ in range(300):
            m = tr.update(rec.buffer)
        e1 = tr.online_error(rec.buffer, n=1024, k=8)
        print("\nbefore:", [f"{v:.4f}" for v in e0.tolist()], "\nafter: ", [f"{v:.4f}" for v in e1.tolist()], "\nlast metrics:", m)
        assert torch.isfinite(e1).all()
        assert e1[0] < 0.5 * e0[0], (e0[0].item(), e1[0].item())
        assert e1[-1] < 0.5 * e0[-1], (e0[-1].item(), e1[-1].item())
        # BPTT gradient through a few imagined steps reaches the action input.
        w = rec.buffer.sample_windows(32, 4)
        a = w.a_exec.clone().requires_grad_(True)
        s = w.states[0]
        for j in range(4):
            s = wm.step(s, a[j], w.xi)
        s.square().sum().backward()
        assert torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0
    finally:
        env.close()

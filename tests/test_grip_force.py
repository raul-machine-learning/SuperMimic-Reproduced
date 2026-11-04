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


def test_pressure_agreement(glove):
    from supermimic.kinematics.glove import NTAXEL

    G = len(ROBOT.grip)
    load = torch.zeros(1, G, NTAXEL)
    load[..., :10] = torch.linspace(5.0, 50.0, 10)  # raw glove units, graded
    zero = torch.zeros_like(load)
    assert torch.allclose(glove.pressure_agreement(zero, zero), torch.ones(1, G))  # free hand pressing nothing
    assert glove.pressure_agreement(zero, load)[0, 0] < 0.05  # glove holds, hand presses nothing
    assert glove.pressure_agreement(load / 20.0, zero)[0, 0] < 0.05  # hand presses, glove free
    # Relative: the same pattern scores high at any strength well above the floor, and the scale is irrelevant.
    same = [glove.pressure_agreement(k * load, load)[0, 0].item() for k in (0.2, 1.0, 5.0)]
    assert min(same) > 0.9 and max(same) - min(same) < 0.06
    # Graded: same taxels, other proportions lose; other taxels lose most.
    flat = (load > 0).float() * load.mean()
    assert glove.pressure_agreement(flat / 20.0, load)[0, 0] < same[1] - 0.1
    assert glove.pressure_agreement(torch.roll(load, 20, -1) / 20.0, load)[0, 0] < 0.1


@requires_sweep
def test_library_pressure_reference():
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.envs.tracking_env_cfg import TRACKED_BODIES

    lib = MotionLibrary(sweep_files(1), TRACKED_BODIES, torch.zeros(ROBOT.nj), device="cpu")
    load = lib.taxel_load
    assert load.shape == (lib.num_frames, len(ROBOT.grip), 129) and (load >= 0).all()
    assert torch.equal(lib.grip_hold, load.sum(-1) > 0)  # holding is detected from the glove pressure
    with np.load(sweep_files(1)[0]) as d:
        sides = ["LR".index(s) for s, _ in ROBOT.grip]
        np.testing.assert_allclose(load.numpy(), d["taxel_load"][:, sides])
    frac = lib.grip_hold.float().mean(0)
    assert ((frac > 0.4) & (frac < 0.8)).all(), frac  # one long hold per hand in the sweep clip
    assert len(torch.unique(load[lib.grip_hold])) > 20  # graded, not on/off


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@requires_sweep
def test_taxels_ride_on_simulated_capsules():
    """The reward's taxels (from the simulated body poses) ride on their own body's collision capsules in the
    simulator, at a constant offset, every step, while random actions move the wrists and fingers."""
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.rewards import sim_glove

    env = make_env(SuperMimicEnvCfgArgs(npz_files=sweep_files(1), num_envs=4, obs_noise=False, reset_noise=False, domain_randomization=False,
                                        action_noise_std=0.0, sampling_mode="start", seed=0), "cuda:0")
    try:
        cmd = env.command_manager.get_term("motion")
        glove, m, robot = sim_glove(cmd), env.sim.mj_model, cmd.robot
        # Collision capsules of each taxel's body: (H, 129, K) geom ids, -1 padded.
        geoms = {b: [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f"robot/{robot.body_names[b]}").id
                     and m.geom_type[g] == 3 and m.geom(g).name.endswith("_collision")] for b in glove.body.unique().tolist()}
        K = max(len(v) for v in geoms.values())
        ids = torch.tensor([[(geoms[b] + [-1] * K)[:K] for b in row] for row in glove.body.tolist()], device=env.device)
        size = torch.as_tensor(m.geom_size, dtype=torch.float32, device=env.device)
        torch.manual_seed(0)
        start = None
        for step in range(150):
            a = torch.clamp(0.5 * torch.randn(env.num_envs, ROBOT.na, device=env.device), -1, 1) if step % 10 == 0 else a
            env.step(a)
            x = glove.taxels_world(robot.data.body_link_pos_w, robot.data.body_link_quat_w)  # (N, H, 129, 3)
            gi = ids.clamp(min=0)
            c = env.sim.data.geom_xpos[:, gi]  # (N, H, 129, K, 3)
            ax = env.sim.data.geom_xmat[:, gi][..., :, 2]
            half, rad = size[gi, 1], size[gi, 0]
            s = ((x[..., None, :] - c) * ax).sum(-1).clamp(-half, half)
            gap = ((x[..., None, :] - (c + s[..., None] * ax)).norm(dim=-1) - rad).abs()
            gap = torch.where(ids >= 0, gap, torch.inf).amin(-1)  # (N, H, 129)
            lo, hi = (gap, gap) if step == 0 else (torch.minimum(lo, gap), torch.maximum(hi, gap))
            start = x if start is None else start
        # On the surface (a few palm taxels sit a fixed ~2 mm off it, from the layout), and rigidly so: no drift.
        assert hi.max() < 2.5e-3 and (hi - lo).max() < 1e-4, ((hi - lo).max(), hi.max())
        assert (hi < 5e-4).float().mean() > 0.95
        moved = (x - start).norm(dim=-1)
        assert moved.median() > 0.02, moved.median()
    finally:
        env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@requires_sweep
def test_mocap_pose_pressure_agreement():
    """Hands and props placed at the mocap pose at mid-grip frames, one physics step: the simulated taxel pressure
    agrees with the glove's at those frames better than with the glove's taxels scrambled. The mean is the ceiling
    the mocap pose itself reaches (about 0.2 per hand on the sweep clip)."""
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.rewards import grip_pressure_agreement_sim, sim_glove, sim_taxel_forces
    from supermimic.envs.state_io import StateIO

    times = [3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]  # s, inside both hands' holds
    env = make_env(SuperMimicEnvCfgArgs(npz_files=sweep_files(1), num_envs=len(times), obs_noise=False, reset_noise=False,
                                        domain_randomization=False, action_noise_std=0.0, sampling_mode="start", seed=0), "cuda:0")
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        frames = lib.clip_start[0] + torch.tensor([round(t * 50) for t in times], device=env.device)
        cmd.forced_frame[:] = frames
        env.reset()
        start = cmd.frame.clone()
        StateIO(env).write(torch.arange(env.num_envs, device=env.device), lib._ref[start].float())
        env.step(torch.zeros(env.num_envs, ROBOT.na, device=env.device))
        assert torch.equal(cmd.frame, start + 1) and lib.grip_hold[cmd.frame].all()
        force = sim_taxel_forces(env, cmd)  # (n, G, 129) N
        matched = grip_pressure_agreement_sim(env, cmd)  # (n, G)
        scrambled = sim_glove(cmd).pressure_agreement(force, cmd.taxel_load.roll(17, -1))
        print("mocap-pose pressure agreement per hand", matched.mean(0).tolist(), "taxels scrambled", scrambled.mean(0).tolist())
        assert ((force.sum(-1) > 0.5).float().mean(0) >= 0.6).all()  # the mocap hands press on their props
        assert (matched.mean(0) > 0.15).all(), matched.mean(0)
        assert (matched.mean(0) > scrambled.mean(0) + 0.05).all(), (matched.mean(0), scrambled.mean(0))
    finally:
        env.close()


@requires_sweep
def test_force_head_fits_logged_states_only():
    from supermimic.data.motion_convert import load_robot_model
    from supermimic.envs.mdp.grip_force import num_bodies
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame
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
    states = torch.stack([s, s])
    loss, m = wm.window_loss(states, a[None], xi, torch.zeros(1, n), [ReferenceFrame(None, None, None, None, r.root_pos_w, r.root_quat_w, r.joint_pos, r.prop_pos_w)],
                             forces=torch.stack([f_true, f_true]))
    assert torch.isfinite(m["force_loss"]) and m["force_mae_n"] > 0
    fl, f_err = wm._force_step(s, a, f_true)
    assert fl.shape == (n,) and f_err.shape == (n,) and torch.isfinite(fl).all()
    fl.sum().backward()
    assert all(p.grad is None or not p.grad.any() for p in wm.dynamics.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in wm.force_head.parameters())

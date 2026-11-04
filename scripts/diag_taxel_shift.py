"""Glove pressure agreement at the mocap pose as the taxel layout is shifted along the hand.

  SM_ROBOT=capsule_hands_props uv run python scripts/diag_taxel_shift.py

Hands and props are placed at the mocap pose at mid-grip frames and stepped once (as
tests/test_grip_force.py::test_mocap_pose_pressure_agreement); the contact forces of the hand segments do not depend on
the layout, so they are simulated once and every shifted layout is scored on them. Shifts are added to the fitted
layout (supermimic.data.glove_taxels.fitted_adjust): ``palm`` moves the five palm rows along the palm (positive:
toward the finger bases), ``fingers`` slides every fingertip pad and phalanx taxel along its segment (positive: toward
the tip). Besides the agreement it reports the share of the glove's load on taxels within 5 mm of the prop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

FINGER_GROUPS = ("thumb_pad", "thumb_base", *(f"{f}_{k}" for f in ("index", "middle", "ring", "pinky") for k in ("pad", "middle", "proximal")))


def main(t0: float = 3.2, t1: float = 10.8, dt: float = 0.2, palm: tuple[float, ...] = tuple(np.round(np.arange(-0.03, 0.0301, 0.005), 3)),
         fingers: tuple[float, ...] = (-0.012, -0.008, -0.004, 0.0, 0.004, 0.008, 0.012)) -> None:
    from sweep_data import sweep_files

    import supermimic.data.glove_taxels as gt
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.grip_force import body_forces
    from supermimic.envs.state_io import StateIO
    from supermimic.kinematics.glove import GloveModel
    from supermimic.robots import ROBOT

    times = np.arange(t0, t1 + 1e-6, dt)
    env = make_env(SuperMimicEnvCfgArgs(npz_files=sweep_files(1), num_envs=len(times), obs_noise=False, reset_noise=False,
                                        domain_randomization=False, action_noise_std=0.0, sampling_mode="start", seed=0), "cuda:0")
    try:
        cmd = env.command_manager.get_term("motion")
        lib, robot = cmd.library, cmd.robot
        cmd.forced_frame[:] = lib.clip_start[0] + torch.tensor([round(t * 50) for t in times], device=env.device)
        env.reset()
        StateIO(env).write(torch.arange(env.num_envs, device=env.device), lib._ref[cmd.frame].float())
        env.step(torch.zeros(env.num_envs, ROBOT.na, device=env.device))
        assert lib.grip_hold[cmd.frame].all()
        bf = body_forces(env)
        bp, bq, pp, pq = (x.clone() for x in (robot.data.body_link_pos_w, robot.data.body_link_quat_w, cmd.sim_prop_pos_w, cmd.sim_prop_quat_w))
        load = cmd.taxel_load.clone()
    finally:
        env.close()

    fitted = gt.fitted_adjust()
    index = lambda names: torch.tensor([robot.body_names.index(n) for n in names], device=bf.device)
    ref_bodies = GloveModel(index, bf.device).force_body_names

    def score(adjust: dict) -> tuple[list[float], list[float]]:
        orig = gt.fitted_adjust
        gt.fitted_adjust = lambda: adjust
        try:
            g = GloveModel(index, bf.device)
        finally:
            gt.fitted_adjust = orig
        # Keep the simulated force bodies' order; a taxel moved onto another body takes that body's force.
        g.slot = torch.stack([torch.tensor([ref_bodies[h].index(robot.body_names[b]) for b in g.body[h].tolist()], device=bf.device)
                              for h in range(len(ROBOT.grip))])
        g.onehot = torch.nn.functional.one_hot(g.slot, len(ref_bodies[0])).float()
        d = g.distances(bp, bq, pp, pq)
        agree = g.pressure_agreement(g.taxel_forces(d, bf), load).mean(0).tolist()
        near = ((load * (d < 0.005)).sum((0, 2)) / load.sum((0, 2))).tolist()
        return agree, near

    def shifted(groups, delta: float) -> dict:
        out = dict(fitted)
        for k in groups:
            a, r = out.get(k, (0.0, 0.0))
            out[k] = (a + delta, r)
        return out

    print(f"mocap pose, {len(times)} frames {t0}-{t1} s; agreement (left broom, right dustpan) and glove load within 5 mm of the prop")
    for name, groups, deltas in (("palm rows, along the palm (+ toward the finger bases)", ("palm",), palm),
                                 ("finger patches, along each segment (+ toward the tip)", FINGER_GROUPS, fingers)):
        print(f"\n{name}")
        for dlt in deltas:
            try:
                a, n = score(shifted(groups, dlt))
            except ValueError:
                print(f"  {dlt * 1000:+5.0f} mm   (taxels off the palm capsules)")
                continue
            print(f"  {dlt * 1000:+5.0f} mm   agreement L {a[0]:.3f}  R {a[1]:.3f}   load near prop L {n[0]:4.0%}  R {n[1]:4.0%}")


if __name__ == "__main__":
    tyro.cli(main)

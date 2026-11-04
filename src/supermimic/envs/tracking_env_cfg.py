"""SuperMimic tracking environment on mjlab (plan Phase 2.3).

Differences from mjlab's BeyondMimic re-implementation (``mjlab.tasks.tracking``):
  * observations: paper Table A.1 proprioception only (no anchor pose, no base lin vel),
    10-frame history (term-major), plus a separate ``goal`` group with K future frames;
  * action: ``NoisyJointPositionAction`` (clean action observed, noise only executed);
  * multi-clip command with return-based reset sampling and clip-end time-outs;
  * rewards: self-collision term removed, joint-limit weight -2;
  * terminations: pelvis height / tilt vs reference, no end-effector term, clip-end time-out;
  * events: no push; friction and encoder bias re-sampled per episode; COM at startup;
  * a ``ReplayRecorder`` streams transitions into a GPU replay buffer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tracking import mdp as track_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from supermimic.data.recorder import ReplayRecorderCfg
from supermimic.envs.actions import NoisyJointPositionActionCfg
from supermimic.envs.mdp import observations as sm_obs
from supermimic.envs.mdp import terminations as sm_term
from supermimic.envs.mdp.commands import MultiClipMotionCommandCfg
from supermimic.envs.mdp import rewards as sm_rew
from supermimic.envs.mdp.rewards import TrackingRewardCfg, motion_hand_joint_position_error_exp
from supermimic.robots import ROBOT

class QuietSimulationCfg(SimulationCfg):
    """SimulationCfg that disables MuJoCo Warp's per-step overflow printf (a kernel build-time constant,
    so it must be set before the CUDA graphs are captured)."""

    def apply_wp_opt(self, wp_opt) -> None:
        super().apply_wp_opt(wp_opt)
        wp_opt.warn_overflow = 0


TRACKED_BODIES = ROBOT.tracked_bodies
ANCHOR_BODY = ROBOT.anchor_body
ROOT_BODY = ROBOT.root_body
FOOT_GEOM_REGEX = ROBOT.foot_geom_regex
PROP_OBS_NOISE = 0.005  # half-width of the uniform noise on the observed prop poses


@dataclass
class SuperMimicEnvCfgArgs:
    npz_files: list[str]
    num_envs: int = 4096
    future_frames: int = 10
    history_length: int = 10
    episode_length_s: float = 10.0
    action_noise_std: float = 0.1
    obs_noise: bool = True
    reset_noise: bool = True
    domain_randomization: bool = True
    sampling_mode: str = "return"  # return | uniform | start
    replay_capacity: int = 1000
    record_transitions: bool = True
    seed: int | None = 0
    # The two termination thresholds (ROBOT.termination): root height / tilt, or wrist position / orientation.
    root_height_threshold: float = ROBOT.termination_thresholds[0]
    root_tilt_threshold: float = ROBOT.termination_thresholds[1]
    joint_limit_weight: float = -2.0
    bin_frames: int = 50
    compact_library: bool = False
    anchor_obs: bool = False  # add BeyondMimic anchor-relative pose (9) to the goal group (plan D5 alt.)
    debug_vis: bool = False  # draw the reference ghost in the viewer (scripts/play.py)
    extra_command_kwargs: dict = field(default_factory=dict)


def make_env_cfg(a: SuperMimicEnvCfgArgs) -> ManagerBasedRlEnvCfg:
    # ---------------------------------------------------------------- observations
    noise = (lambda lo, hi: Unoise(n_min=lo, n_max=hi)) if a.obs_noise else (lambda lo, hi: None)
    policy_terms = {
        "joint_pos": ObservationTermCfg(func=base_mdp.joint_pos_rel, params={"biased": True}, noise=noise(-0.01, 0.01)),
        "joint_vel": ObservationTermCfg(func=base_mdp.joint_vel_rel, noise=noise(-0.5, 0.5)),
        "base_ang_vel": ObservationTermCfg(func=base_mdp.base_ang_vel, noise=noise(-0.2, 0.2)),
        "projected_gravity": ObservationTermCfg(func=base_mdp.projected_gravity, noise=noise(-0.05, 0.05)),
        "actions": ObservationTermCfg(func=base_mdp.last_action),
    }
    if ROBOT.props:
        policy_terms["prop_pose"] = ObservationTermCfg(func=sm_obs.prop_pose, params={"command_name": "motion"}, noise=noise(-PROP_OBS_NOISE, PROP_OBS_NOISE))
    observations = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=a.obs_noise,
            history_length=a.history_length,
            flatten_history_dim=True,
        ),
        "goal": ObservationGroupCfg(
            terms={
                "future_frames": ObservationTermCfg(func=sm_obs.goal_future_frames, params={"command_name": "motion"}),
                **(
                    {
                        "anchor_pos": ObservationTermCfg(func=track_mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=noise(-0.25, 0.25)),
                        "anchor_ori": ObservationTermCfg(func=track_mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=noise(-0.05, 0.05)),
                    }
                    if a.anchor_obs
                    else {}
                ),
            },
            concatenate_terms=True,
            enable_corruption=a.obs_noise,
        ),
    }

    # ---------------------------------------------------------------- actions
    actions = {
        "joint_pos": NoisyJointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=ROBOT.action_scale_fn(),
            use_default_offset=True,
            noise_std=a.action_noise_std,
            reference_offset=ROBOT.residual_actions,
        )
    }

    # ---------------------------------------------------------------- commands
    pose_range = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01), "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2)}
    vel_range = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2), "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78)}
    if ROBOT.fixed_root:
        pose_range, vel_range = {}, {}
    commands = {
        "motion": MultiClipMotionCommandCfg(
            npz_files=[str(f) for f in a.npz_files],
            anchor_body_name=ANCHOR_BODY,
            root_body_name=ROOT_BODY,
            body_names=TRACKED_BODIES,
            entity_name="robot",
            future_frames=a.future_frames,
            bin_frames=a.bin_frames,
            pose_range=pose_range if a.reset_noise else {},
            velocity_range=vel_range if a.reset_noise else {},
            joint_position_range=(-0.1, 0.1) if a.reset_noise else (0.0, 0.0),
            sampling_mode=a.sampling_mode,  # type: ignore[arg-type]
            compact_library=a.compact_library,
            debug_vis=a.debug_vis,
            **a.extra_command_kwargs,
        )
    }

    # ---------------------------------------------------------------- events (plan D8)
    events: dict[str, EventTermCfg] = {}
    if a.domain_randomization:
        events["base_com"] = EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=(ROBOT.com_body,)),
                "operation": "add",
                "ranges": {0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
            },
        )
        events["encoder_bias"] = EventTermCfg(
            mode="reset",
            func=dr.encoder_bias,
            params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)},
        )
        events["foot_friction"] = EventTermCfg(
            mode="reset",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_REGEX),
                "operation": "abs",
                "ranges": ROBOT.contact_friction_range,
                "shared_random": True,
            },
        )
    else:
        # Evaluation protocol: foot friction fixed at 1.0, no other randomization.
        events["foot_friction"] = EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_REGEX),
                "operation": "abs",
                "ranges": (1.0, 1.0),
                "shared_random": True,
            },
        )

    # ---------------------------------------------------------------- rewards (Table A.2)
    # Fingers rest against their limits (open / closed hand), so only the body joints are penalised there.
    limit_joints = (".*",) if ROBOT.limit_joint_names == ROBOT.joint_names else ROBOT.limit_joint_names
    rc = TrackingRewardCfg()
    rewards = {
        "motion_global_root_pos": RewardTermCfg(func=track_mdp.motion_global_anchor_position_error_exp, weight=rc.w_root_pos, params={"command_name": "motion", "std": rc.std_root_pos}),
        "motion_global_root_ori": RewardTermCfg(func=track_mdp.motion_global_anchor_orientation_error_exp, weight=rc.w_root_ori, params={"command_name": "motion", "std": rc.std_root_ori}),
        "motion_body_pos": RewardTermCfg(func=track_mdp.motion_relative_body_position_error_exp, weight=rc.w_body_pos, params={"command_name": "motion", "std": rc.std_body_pos}),
        "motion_body_ori": RewardTermCfg(func=track_mdp.motion_relative_body_orientation_error_exp, weight=rc.w_body_ori, params={"command_name": "motion", "std": rc.std_body_ori}),
        "motion_body_lin_vel": RewardTermCfg(func=track_mdp.motion_global_body_linear_velocity_error_exp, weight=rc.w_body_lin_vel, params={"command_name": "motion", "std": rc.std_body_lin_vel}),
        "motion_body_ang_vel": RewardTermCfg(func=track_mdp.motion_global_body_angular_velocity_error_exp, weight=rc.w_body_ang_vel, params={"command_name": "motion", "std": rc.std_body_ang_vel}),
        "action_rate_l2": RewardTermCfg(func=base_mdp.action_rate_l2, weight=rc.w_action_rate),
        "joint_limit": RewardTermCfg(func=base_mdp.joint_pos_limits, weight=a.joint_limit_weight, params={"asset_cfg": SceneEntityCfg("robot", joint_names=limit_joints)}),
    }
    if ROBOT.grip:
        for name, func in (("motion_body_pos", sm_rew.motion_body_position_released_exp), ("motion_body_ori", sm_rew.motion_body_orientation_released_exp),
                           ("motion_body_lin_vel", sm_rew.motion_body_lin_vel_released_exp), ("motion_body_ang_vel", sm_rew.motion_body_ang_vel_released_exp)):
            rewards[name] = RewardTermCfg(func=func, weight=rewards[name].weight, params=rewards[name].params)
    if ROBOT.hand_joint_names:
        rewards["motion_hand_joint_pos"] = RewardTermCfg(
            func=motion_hand_joint_position_error_exp, weight=rc.w_hand_joint_pos,
            params={"command_name": "motion", "std": rc.std_hand_joint_pos, "joint_ids": ROBOT.joint_indices(ROBOT.hand_joint_names)},
        )
    if ROBOT.props:
        rewards["motion_prop_pos"] = RewardTermCfg(func=sm_rew.motion_prop_position_error_exp, weight=rc.w_prop_pos, params={"command_name": "motion", "std": rc.std_prop_pos})
        rewards["motion_prop_ori"] = RewardTermCfg(func=sm_rew.motion_prop_orientation_error_exp, weight=rc.w_prop_ori, params={"command_name": "motion", "std": rc.std_prop_ori})
    if ROBOT.grip:
        rewards["grip_pressure"] = RewardTermCfg(func=sm_rew.grip_pressure_agreement, weight=rc.w_grip, params={"command_name": "motion"})

    # ---------------------------------------------------------------- terminations
    terminations = {
        "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
        "clip_end": TerminationTermCfg(func=sm_term.clip_end, params={"command_name": "motion"}, time_out=True),
        "bad_sim_state": TerminationTermCfg(func=sm_term.bad_sim_state),
    }
    if ROBOT.termination == "wrists":
        terminations["wrist_pos"] = TerminationTermCfg(func=sm_term.bad_wrist_pos, params={"command_name": "motion", "threshold": a.root_height_threshold})
        terminations["wrist_ori"] = TerminationTermCfg(func=sm_term.bad_wrist_ori, params={"command_name": "motion", "threshold": a.root_tilt_threshold})
    else:
        terminations["root_height"] = TerminationTermCfg(func=sm_term.bad_root_height, params={"command_name": "motion", "threshold": a.root_height_threshold})
        terminations["root_tilt"] = TerminationTermCfg(func=sm_term.bad_root_tilt, params={"command_name": "motion", "threshold": a.root_tilt_threshold})
    if ROBOT.props:
        terminations["prop_pos"] = TerminationTermCfg(func=sm_term.bad_prop_pos, params={"command_name": "motion", "threshold": ROBOT.prop_drop_threshold})

    entities = {"robot": ROBOT.entity_cfg_fn()}
    if ROBOT.props:
        from supermimic.envs.props import PropSceneCfg, prop_entity_cfgs

        entities.update(prop_entity_cfgs(PropSceneCfg.from_clip(a.npz_files[0]), ROBOT.props))

    recorders = {}
    if a.record_transitions:
        recorders["replay"] = ReplayRecorderCfg(params={"capacity": a.replay_capacity, "foot_geom_regex": FOOT_GEOM_REGEX, "com_body_name": ROBOT.com_body})

    sensors = ()
    if ROBOT.grip:
        from supermimic.envs.mdp.grip_force import sensor_cfgs

        sensors = sensor_cfgs()

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=a.num_envs, entities=entities,
                       env_spacing=0.0 if ROBOT.fixed_root else 2.0, sensors=sensors),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        recorders=recorders,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name=ROBOT.anchor_body, distance=2.8, fovy=55.0, elevation=-5.0, azimuth=120.0),
        sim=QuietSimulationCfg(nconmax=ROBOT.nconmax, njmax=ROBOT.njmax, mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)),
        decimation=4,
        episode_length_s=a.episode_length_s,
        seed=a.seed,
    )
    return cfg


def make_env(a: SuperMimicEnvCfgArgs, device: str = "cuda:0") -> ManagerBasedRlEnv:
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    return ManagerBasedRlEnv(make_env_cfg(a), device=device)

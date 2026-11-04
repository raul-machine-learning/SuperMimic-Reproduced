"""Robot variants and the sizes derived from them.

* ``g1`` (default): mjlab's Unitree G1, 29 hinge joints, all actuated.
* ``g1_inspire``: the same G1 with Inspire RH56 DFQ five-finger hands in place of the rubber hands. Each hand
  has 12 hinge joints, 6 of them actuated (thumb yaw, thumb pitch, the proximal joint of each finger); the other
  6 follow through the URDF ``<mimic>`` couplings, kept as MuJoCo joint equalities. 53 hinges, 41 actuators.
* ``g1_inspire_sweep``: ``g1_inspire`` with two free props (broom, dustpan) and a table in the scene, for the
  sweep clip with physical grasps; the hand collision geoms get sliding and torsional friction (condim 4).
* ``capsule_hands``: two capsule hands built from the sweep mocap skeleton (scripts/assets/build_capsule_hands.py),
  no robot body. The root ``rig`` is a free body welded to the world, so the floating-root state layout is kept;
  each hand hangs from it through 3 world slides and 3 hinges, then 21 finger hinges. 54 joints, all actuated.
* ``capsule_hands_props``: ``capsule_hands`` with the broom (held by the left hand) and the dustpan (right hand)
  as free props and the table, plus the glove-pattern grip reward.

The two capsule-hand robots come in two segment shapes, chosen by ``SM_HAND_SHAPE``: ``capsule`` (default) or
``box`` (every capsule replaced by a box of the same span with a square cross-section, flat on the palm side). Each
shape has its own assets (``assets/<shape>_hands``), hand clips (``data/hands`` / ``data/hands_box``), rig clips
(``data/npz/<shape>_hands``) and grasp offsets.

The state, observation and goal layouts are sized by the number of hinge joints (``nj``) and of props, the
action by the number of actuated joints (``na``). The active robot is fixed per process by the ``SM_ROBOT``
environment variable, since those sizes are module-level constants throughout the package.

The state is the robot block ``[qpos | qvel]`` followed by one 13-d block per prop
``[pos 3, quat wxyz 4, lin vel world 3, ang vel body 3]`` (the prop's free-joint qpos and qvel).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import mujoco
import numpy as np

if TYPE_CHECKING:
    from mjlab.entity import EntityCfg

ASSETS = Path(__file__).resolve().parent / "assets"
SLIDE_SCALE = 0.2  # m per rad: hand travel is ~0.2x the joint angle range in the state / noise / loss scales

G1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

G1_TRACKED_BODIES: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)


def _hand_joints(side: str) -> tuple[str, ...]:
    """One hand's hinges in MuJoCo (depth-first) order."""
    return tuple(
        f"{side}_{n}_joint"
        for n in (
            "thumb_proximal_yaw", "thumb_proximal_pitch", "thumb_intermediate", "thumb_distal",
            "index_proximal", "index_intermediate", "middle_proximal", "middle_intermediate",
            "ring_proximal", "ring_intermediate", "pinky_proximal", "pinky_intermediate",
        )
    )


# follower -> (driver, multiplier) of the URDF <mimic> couplings (offsets are all zero).
INSPIRE_MIMIC: dict[str, tuple[str, float]] = {}
for _s in "LR":
    INSPIRE_MIMIC[f"{_s}_thumb_intermediate_joint"] = (f"{_s}_thumb_proximal_pitch_joint", 1.6)
    INSPIRE_MIMIC[f"{_s}_thumb_distal_joint"] = (f"{_s}_thumb_proximal_pitch_joint", 2.4)
    for _f in ("index", "middle", "ring", "pinky"):
        INSPIRE_MIMIC[f"{_s}_{_f}_intermediate_joint"] = (f"{_s}_{_f}_proximal_joint", 1.0)

_LW = G1_JOINT_NAMES.index("left_wrist_yaw_joint") + 1
INSPIRE_JOINT_NAMES = G1_JOINT_NAMES[:_LW] + _hand_joints("L") + G1_JOINT_NAMES[_LW:] + _hand_joints("R")
INSPIRE_HAND_ACTUATED = tuple(j for j in _hand_joints("L") + _hand_joints("R") if j not in INSPIRE_MIMIC)

# URDF inertial of *_wrist_yaw_link without a hand (mjlab's g1.xml folds the rubber hand into it).
_WRIST_YAW_INERTIAL = {
    "left": dict(mass=0.08457647, ipos=[0.02200382, 0.00049485, 0.00053861],
                 iquat=[0.68276135, -0.12133658, -0.13511889, 0.7077127], inertia=[5.97564e-05, 5.09807e-05, 3.75684e-05]),
    "right": dict(mass=0.08457647, ipos=[0.02200382, -0.00049485, 0.00053861],
                  iquat=[0.7077127, -0.13511889, -0.12133658, 0.68276135], inertia=[5.97564e-05, 5.09807e-05, 3.75684e-05]),
}
# Both thumb followers hit their upper limits (0.8 / 1.2) at pitch 0.5 and their lower limit 0 at pitch 0.
INSPIRE_THUMB_PITCH_RANGE = (0.0, 0.5)
# Relaxed hand: used as the action offset and the observation origin of the hand joints.
INSPIRE_HAND_DEFAULT = {"thumb_proximal_yaw": 0.6, "thumb_proximal_pitch": 0.2, "proximal": 0.4}
INSPIRE_HAND_ACTION_SCALE = 0.6  # rad per unit action
# Finger drive: the URDF effort limit, and a 10 Hz / damping-ratio-2 position loop on the armature, like
# mjlab's G1 actuators.
INSPIRE_HAND_EFFORT = 1.0
INSPIRE_HAND_ARMATURE = 1.0e-3
INSPIRE_FOLLOWER_ARMATURE = 1.0e-4


@dataclass(frozen=True)
class RobotSpec:
    name: str
    joint_names: tuple[str, ...]  # every hinge, in qpos order
    actuated_joint_names: tuple[str, ...]  # in qpos order (= mjlab action order)
    tracked_bodies: tuple[str, ...]
    anchor_body: str
    root_body: str
    foot_geom_regex: str
    limit_joint_names: tuple[str, ...]  # joints under the soft joint-limit penalty
    hand_joint_names: tuple[str, ...]  # joints scored by the hand-tracking reward (empty: no such term)
    mimic: tuple[tuple[str, str, float], ...]  # (follower, driver, multiplier): follower = multiplier * driver
    entity_cfg_fn: Callable[[], EntityCfg]
    action_scale_fn: Callable[[], dict[str, float]]
    spec_fn: Callable[[], mujoco.MjSpec]
    nconmax: int
    njmax: int
    props: tuple[str, ...] = ()  # free rigid props simulated with the robot (supermimic.envs.props)
    com_body: str = "torso_link"  # body whose COM is randomised (xi's COM offset)
    contact_friction_range: tuple[float, float] = (0.3, 1.2)  # randomised friction of the foot_geom_regex geoms
    # "root": root height / tilt against the reference (the paper's); "wrists": wrist position / orientation error,
    # read from the wrist joints. Thresholds in that order (m, unitless tilt) or (m, rad).
    termination: str = "root"
    termination_thresholds: tuple[float, float] = (0.25, 0.8)
    wrist_joints: tuple[tuple[str, ...], ...] = ()  # per hand: slides x, y, z (world), hinges y, x, z (intrinsic)
    fixed_root: bool = False  # root welded to the world: no root reset noise, all env origins at the world origin
    reward_overrides: tuple[tuple[str, float], ...] = ()  # TrackingRewardCfg field -> value
    slide_joint_names: tuple[str, ...] = ()  # in metres: scaled by SLIDE_SCALE wherever joints share one scale
    # (hand side, prop) pairs scored by the glove-pattern grip reward (supermimic.kinematics.glove).
    grip: tuple[tuple[str, str], ...] = ()
    prop_drop_threshold: float = 0.15  # m: a prop this far from its reference ends the episode
    # Joint set-points are the next reference frame's joint positions plus the scaled action (instead of the
    # default pose plus the scaled action; supermimic.envs.actions).
    residual_actions: bool = False
    # Grasp offsets (joint regex -> rad) added to a gripping hand's set-points while its glove holds, switched on
    # grip_anticipate frames early; annealed to 0 during training (supermimic.envs.mdp.grip_assist).
    grip_assist: tuple[tuple[str, float], ...] = ()
    grip_anticipate: int = 10

    @property
    def nj(self) -> int:
        return len(self.joint_names)

    @property
    def joint_scale(self) -> tuple[float, ...]:
        """Per joint, the size of 1 rad in its own unit (SLIDE_SCALE m for slides)."""
        return tuple(SLIDE_SCALE if j in self.slide_joint_names else 1.0 for j in self.joint_names)

    @property
    def na(self) -> int:
        return len(self.actuated_joint_names)

    @property
    def nq(self) -> int:
        return 7 + self.nj

    @property
    def nv(self) -> int:
        return 6 + self.nj

    @property
    def num_props(self) -> int:
        return len(self.props)

    @property
    def robot_state_dim(self) -> int:
        """[root pos 3, root quat 4, joint pos nj, root lin vel 3, root ang vel 3, joint vel nj]."""
        return self.nq + self.nv

    @property
    def state_dim(self) -> int:
        """Robot block, then [pos 3, quat 4, lin vel 3, ang vel 3] per prop."""
        return self.robot_state_dim + 13 * self.num_props

    @property
    def feat_dim(self) -> int:
        """World-model input features: robot block, then [pos 3, rot6d 6, lin vel 3, ang vel 3] per prop."""
        return self.robot_state_dim + 15 * self.num_props

    @property
    def delta_dim(self) -> int:
        return 12 + 2 * self.nj + 12 * self.num_props

    @property
    def xi_dim(self) -> int:
        """Foot friction 1 + encoder bias nj + torso COM offset 3."""
        return 4 + self.nj

    @property
    def obs_frame_dim(self) -> int:
        """joint pos nj, joint vel nj, base ang vel 3, projected gravity 3, last action na, prop pose in the root frame 9 per prop."""
        return 2 * self.nj + 6 + self.na + 9 * self.num_props

    @property
    def goal_frame_dim(self) -> int:
        """joint pos - q0 nj, joint vel nj, gravity 3, ang vel 3, lin vel 3, height 1, prop pose in the root frame 9 per
        prop, glove holding flag 1 per gripping hand."""
        return 2 * self.nj + 10 + 9 * self.num_props + len(self.grip)

    def action_scale_vector(self) -> tuple[float, ...]:
        """Per actuated joint, its action scale (the first matching pattern of ``action_scale_fn``)."""
        import re

        scales = self.action_scale_fn()
        return tuple(next(v for k, v in scales.items() if re.fullmatch(k, j)) for j in self.actuated_joint_names)

    def joint_indices(self, names: tuple[str, ...]) -> list[int]:
        return [self.joint_names.index(n) for n in names]

    def split_state(self, s):
        """-> root_pos 3, root_quat 4, joint_pos nj, root_lin_vel_w 3, root_ang_vel 3, joint_vel nj (last axis)."""
        n = self.nj
        return s[..., 0:3], s[..., 3:7], s[..., 7 : 7 + n], s[..., 7 + n : 10 + n], s[..., 10 + n : 13 + n], s[..., 13 + n : 13 + 2 * n]

    def split_props(self, s):
        """-> pos (…,P,3), quat (…,P,4), lin_vel_w (…,P,3), ang_vel (…,P,3) of the prop blocks (torch)."""
        b = s[..., self.robot_state_dim :].reshape(*s.shape[:-1], self.num_props, 13)
        return b[..., 0:3], b[..., 3:7], b[..., 7:10], b[..., 10:13]

    def mj_model(self) -> mujoco.MjModel:
        """Standalone CPU model (no mjlab actuators); checks the joint order."""
        return _compiled(self.name)


# ----------------------------------------------------------------------------- G1


def _g1_spec() -> mujoco.MjSpec:
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

    return get_spec()


def _g1_entity_cfg() -> EntityCfg:
    from mjlab.asset_zoo.robots import get_g1_robot_cfg

    return get_g1_robot_cfg()


def _g1_action_scale() -> dict[str, float]:
    from mjlab.asset_zoo.robots import G1_ACTION_SCALE

    return dict(G1_ACTION_SCALE)


# ----------------------------------------------------------------------------- G1 + Inspire


def _inspire_spec() -> mujoco.MjSpec:
    spec = _g1_spec()
    hands = mujoco.MjSpec.from_file(str(ASSETS / "inspire_hands/inspire_hands.xml"))
    for side, s in (("left", "L"), ("right", "R")):
        wrist = spec.body(f"{side}_wrist_yaw_link")
        for g in list(wrist.geoms):
            if g.name == f"{side}_hand_collision" or g.meshname == f"{side}_rubber_hand":
                spec.delete(g)
        w = _WRIST_YAW_INERTIAL[side]
        wrist.mass, wrist.ipos, wrist.iquat, wrist.inertia = w["mass"], w["ipos"], w["iquat"], w["inertia"]
        wrist.explicitinertial = True
        base = hands.body(f"{s}_hand_base_link")
        frame = wrist.add_frame()
        frame.attach_body(base, "", "")
        hand_bodies = [f"{s}_hand_base_link"] + [b.name for b in spec.body(f"{s}_hand_base_link").find_all(mujoco.mjtObj.mjOBJ_BODY)]
        # Phalanges of one hand never collide with each other (their primitives overlap at the knuckles).
        for i, b1 in enumerate(hand_bodies):
            for b2 in hand_bodies[i + 1 :]:
                spec.add_exclude(bodyname1=b1, bodyname2=b2)
    for j in spec.joints:
        if j.name.endswith("thumb_proximal_pitch_joint"):
            j.range = list(INSPIRE_THUMB_PITCH_RANGE)
        if j.name in INSPIRE_MIMIC:
            j.armature = INSPIRE_FOLLOWER_ARMATURE
    return spec


def _inspire_hand_default() -> dict[str, float]:
    out: dict[str, float] = {}
    for s in "LR":
        out[f"{s}_thumb_proximal_yaw_joint"] = INSPIRE_HAND_DEFAULT["thumb_proximal_yaw"]
        out[f"{s}_thumb_proximal_pitch_joint"] = INSPIRE_HAND_DEFAULT["thumb_proximal_pitch"]
        for f in ("index", "middle", "ring", "pinky"):
            out[f"{s}_{f}_proximal_joint"] = INSPIRE_HAND_DEFAULT["proximal"]
    for follower, (driver, mult) in INSPIRE_MIMIC.items():
        out[follower] = mult * out[driver]
    return out


def _inspire_entity_cfg() -> EntityCfg:
    import math

    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as g1c
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

    wn = 10 * 2.0 * math.pi
    hand = BuiltinPositionActuatorCfg(
        target_names_expr=INSPIRE_HAND_ACTUATED,
        stiffness=INSPIRE_HAND_ARMATURE * wn**2,
        damping=2.0 * 2.0 * INSPIRE_HAND_ARMATURE * wn,
        effort_limit=INSPIRE_HAND_EFFORT,
        armature=INSPIRE_HAND_ARMATURE,
    )
    init = g1c.KNEES_BENT_KEYFRAME
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=init.pos, joint_pos={**init.joint_pos, **_inspire_hand_default()}, joint_vel={".*": 0.0}),
        collisions=(g1c.FULL_COLLISION,),
        spec_fn=_inspire_spec,
        articulation=EntityArticulationInfoCfg(actuators=(*g1c.G1_ARTICULATION.actuators, hand), soft_joint_pos_limit_factor=0.9),
    )


def _inspire_action_scale() -> dict[str, float]:
    return {**_g1_action_scale(), **{j: INSPIRE_HAND_ACTION_SCALE for j in INSPIRE_HAND_ACTUATED}}


# Hands that grasp: sliding friction 1 and torsional friction 5 mm (about a fingertip contact patch). mjlab's
# G1 collision config makes every non-foot geom frictionless (condim 1).
INSPIRE_HAND_GEOMS = r"^[LR]_.*_collision$"
INSPIRE_HAND_FRICTION = (1.0, 0.005, 0.0001)


def _inspire_sweep_entity_cfg() -> EntityCfg:
    from mjlab.utils.spec_config import CollisionCfg

    feet = _G1_FEET
    cfg = _inspire_entity_cfg()
    cfg.collisions = (
        CollisionCfg(
            geom_names_expr=(".*_collision",),
            contype=1,
            conaffinity=1,
            condim={feet: 3, INSPIRE_HAND_GEOMS: 4, ".*_collision": 1},
            priority={feet: 1, ".*": 0},
            friction={feet: (0.6,), INSPIRE_HAND_GEOMS: INSPIRE_HAND_FRICTION},
        ),
    )
    return cfg


# ----------------------------------------------------------------------------- capsule hands

HAND_SHAPE = os.environ.get("SM_HAND_SHAPE", "capsule")
assert HAND_SHAPE in ("capsule", "box"), f"SM_HAND_SHAPE must be capsule or box, not {HAND_SHAPE!r}"
HAND_ASSETS = ASSETS / f"{HAND_SHAPE}_hands"
HANDS_XML = HAND_ASSETS / f"{HAND_SHAPE}_hands.xml"
CAPSULE_RIG_XML = HAND_ASSETS / f"{HAND_SHAPE}_rig.xml"
_REPO = Path(__file__).resolve().parents[2]
HAND_DATA = _REPO / ("data/hands" if HAND_SHAPE == "capsule" else f"data/hands_{HAND_SHAPE}")
HAND_CLIPS = _REPO / f"data/npz/{HAND_SHAPE}_hands"


def _xml_joint_names(path: Path) -> tuple[str, ...]:
    """Non-free joints in document order (= MuJoCo's depth-first order)."""
    import xml.etree.ElementTree as ET

    return tuple(j.get("name") for j in ET.parse(path).getroot().iter("joint") if j.get("type") != "free")


CAPSULE_JOINT_NAMES = _xml_joint_names(CAPSULE_RIG_XML) if CAPSULE_RIG_XML.exists() else ()
CAPSULE_WRIST_JOINTS = tuple(tuple(f"{s}_wrist_{a}" for a in ("x", "y", "z", "ry", "rx", "rz")) for s in "LR")
CAPSULE_SLIDES = tuple(j for w in CAPSULE_WRIST_JOINTS for j in w[:3])
CAPSULE_FINGER_JOINTS = tuple(j for j in CAPSULE_JOINT_NAMES if "_wrist_" not in j)
CAPSULE_DIGITS = ("thumb", "index", "middle", "ring", "pinky")
CAPSULE_TRACKED_BODIES = ("rig",) + tuple(
    b for s in "LR" for b in (f"{s}_hand", *(f"{s}_{d}{k}" for d in CAPSULE_DIGITS for k in (1, 3)))
)
CAPSULE_HAND_GEOMS = r"^[LR]_.*_collision$"
CAPSULE_ROOT_MASS = 1000.0  # kg (gravity-compensated); inertia CAPSULE_ROOT_MASS kg m^2 about each axis
# Position loops at 10 Hz, damping ratio 1, on each joint's inertia at the mean pose (hand mass + armature);
# (inertia, effort limit) per joint group.
CAPSULE_DRIVE = {
    r"^[LR]_wrist_[xyz]$": (0.45, 60.0),
    r"^[LR]_wrist_r[xyz]$": (5.0e-3, 10.0),
    r"^[LR]_(index|middle|ring|pinky)1_flex$": (2.0e-3, 2.0),
    r"^[LR]_(index|middle|ring|pinky)1_spread$": (2.0e-3, 1.0),
    r"^[LR]_(index|middle|ring|pinky)2_flex$": (2.0e-3, 1.0),
    r"^[LR]_(index|middle|ring|pinky)3_flex$": (2.0e-3, 0.5),
    r"^[LR]_thumb1_(flex|spread)$": (2.0e-3, 2.0),
    r"^[LR]_thumb2_flex$": (2.0e-3, 1.5),
    r"^[LR]_thumb2_spread$": (2.0e-3, 1.0),
    r"^[LR]_thumb3_flex$": (2.0e-3, 0.8),
}
CAPSULE_ACTION_SCALE = {r"^[LR]_wrist_[xyz]$": 0.1, r"^[LR]_wrist_r[xyz]$": 0.5, r"^[LR]_(?!wrist_).*$": 0.5}


def _capsule_rig_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(CAPSULE_RIG_XML))
    # MjSpec.to_xml drops explicit inertials: restore those of the geometry-free bodies (root, wrist chain). The
    # root is heavy so that the wrist drives' reactions do not move it against the (soft) weld.
    for b in spec.bodies:
        if b.name and not b.geoms:
            b.explicitinertial = True
            b.mass, b.inertia = (CAPSULE_ROOT_MASS, [CAPSULE_ROOT_MASS] * 3) if b.name == "rig" else (1e-3, [1e-7] * 3)
    weld = spec.equality("rig_weld")
    weld.data = [0.0] * 3 + [0.0] * 3 + [1.0, 0.0, 0.0, 0.0] + [1.0]  # anchor, relpose pos, relpose quat, torquescale
    return spec


@cache
def _capsule_default_joint_pos() -> dict[str, float]:
    """Wrists at the clip's mean pose (0); fingers at 0 moved 10 % inside their range."""
    m = _capsule_rig_spec().compile()
    out = {}
    for j in CAPSULE_JOINT_NAMES:
        lo, hi = m.joint(j).range
        out[j] = float(np.clip(0.0, lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo)))
    return out


def _capsule_entity_cfg() -> EntityCfg:
    import math

    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

    wn = 10 * 2.0 * math.pi
    drives = tuple(
        BuiltinPositionActuatorCfg(target_names_expr=(expr,), stiffness=inertia * wn**2, damping=2.0 * inertia * wn, effort_limit=effort)
        for expr, (inertia, effort) in CAPSULE_DRIVE.items()
    )
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos=_capsule_default_joint_pos(), joint_vel={".*": 0.0}),
        spec_fn=_capsule_rig_spec,
        articulation=EntityArticulationInfoCfg(actuators=drives, soft_joint_pos_limit_factor=0.9),
    )


def _capsule_action_scale() -> dict[str, float]:
    return dict(CAPSULE_ACTION_SCALE)


# ----------------------------------------------------------------------------- registry

_G1_FEET = r"^(left|right)_foot[1-7]_collision$"

ROBOTS: dict[str, RobotSpec] = {
    "g1": RobotSpec(
        name="g1",
        joint_names=G1_JOINT_NAMES,
        actuated_joint_names=G1_JOINT_NAMES,
        tracked_bodies=G1_TRACKED_BODIES,
        anchor_body="torso_link",
        root_body="pelvis",
        foot_geom_regex=_G1_FEET,
        limit_joint_names=G1_JOINT_NAMES,
        hand_joint_names=(),
        mimic=(),
        entity_cfg_fn=_g1_entity_cfg,
        action_scale_fn=_g1_action_scale,
        spec_fn=_g1_spec,
        nconmax=35,
        njmax=250,
    ),
    "g1_inspire": RobotSpec(
        name="g1_inspire",
        joint_names=INSPIRE_JOINT_NAMES,
        actuated_joint_names=tuple(j for j in INSPIRE_JOINT_NAMES if j not in INSPIRE_MIMIC),
        tracked_bodies=G1_TRACKED_BODIES,
        anchor_body="torso_link",
        root_body="pelvis",
        foot_geom_regex=_G1_FEET,
        limit_joint_names=G1_JOINT_NAMES,
        hand_joint_names=INSPIRE_HAND_ACTUATED,
        mimic=tuple((f, d, m) for f, (d, m) in INSPIRE_MIMIC.items()),
        entity_cfg_fn=_inspire_entity_cfg,
        action_scale_fn=_inspire_action_scale,
        spec_fn=_inspire_spec,
        nconmax=60,
        njmax=400,
    ),
    "g1_inspire_sweep": RobotSpec(
        name="g1_inspire_sweep",
        joint_names=INSPIRE_JOINT_NAMES,
        actuated_joint_names=tuple(j for j in INSPIRE_JOINT_NAMES if j not in INSPIRE_MIMIC),
        tracked_bodies=G1_TRACKED_BODIES,
        anchor_body="torso_link",
        root_body="pelvis",
        foot_geom_regex=_G1_FEET,
        limit_joint_names=G1_JOINT_NAMES,
        hand_joint_names=INSPIRE_HAND_ACTUATED,
        mimic=tuple((f, d, m) for f, (d, m) in INSPIRE_MIMIC.items()),
        entity_cfg_fn=_inspire_sweep_entity_cfg,
        action_scale_fn=_inspire_action_scale,
        spec_fn=_inspire_spec,
        nconmax=200,
        njmax=1200,
        props=("broom", "dustpan"),
    ),
    "capsule_hands": RobotSpec(
        name="capsule_hands",
        joint_names=CAPSULE_JOINT_NAMES,
        actuated_joint_names=CAPSULE_JOINT_NAMES,
        tracked_bodies=CAPSULE_TRACKED_BODIES,
        anchor_body="rig",
        root_body="rig",
        foot_geom_regex=CAPSULE_HAND_GEOMS,
        limit_joint_names=CAPSULE_JOINT_NAMES,
        hand_joint_names=CAPSULE_FINGER_JOINTS,
        mimic=(),
        entity_cfg_fn=_capsule_entity_cfg,
        action_scale_fn=_capsule_action_scale,
        spec_fn=_capsule_rig_spec,
        nconmax=64,
        njmax=300,
        com_body="rig",
        contact_friction_range=(0.7, 1.2),
        termination="wrists",
        termination_thresholds=(0.10, 0.8),
        wrist_joints=CAPSULE_WRIST_JOINTS,
        fixed_root=True,
        reward_overrides=(("std_body_pos", 0.05), ("std_body_lin_vel", 0.5)),
        slide_joint_names=CAPSULE_SLIDES,
        residual_actions=True,
    ),
}


def _grasp_offsets(S: str, flex1: float, flex2: float, flex3: float, spread: float, thumb1_flex: float, thumb1_spread: float,
                   thumb2_flex: float, thumb2_spread: float, thumb3_flex: float) -> tuple[tuple[str, float], ...]:
    fingers = "(index|middle|ring|pinky)"
    return (
        (rf"{S}_{fingers}1_flex", flex1), (rf"{S}_{fingers}2_flex", flex2), (rf"{S}_{fingers}3_flex", flex3), (rf"{S}_{fingers}1_spread", spread),
        (rf"{S}_thumb1_flex", thumb1_flex), (rf"{S}_thumb1_spread", thumb1_spread), (rf"{S}_thumb2_flex", thumb2_flex),
        (rf"{S}_thumb2_spread", thumb2_spread), (rf"{S}_thumb3_flex", thumb3_flex),
    )


ROBOTS["capsule_hands_props"] = replace(
    ROBOTS["capsule_hands"], name="capsule_hands_props", props=("broom", "dustpan"), grip=(("L", "broom"), ("R", "dustpan")),
    nconmax=96, njmax=600,  # the clip's reference poses need at most 27 contacts and 168 constraint rows
    # From scripts/diag_grasp_search.py: replayed on the reference without perturbations, these carry the broom (2.3 cm
    # mean error, 7 cm lift) and the dustpan (about 1.5 cm, 23 cm lift); the dustpan grasp was refined under the
    # training perturbations (--robust). The broom grip needs the thumb opposed and the fingers adducted.
    # Box hands: 1.2 cm / 6 cm lift on the broom, 0.8 cm / 25 cm on the dustpan.
    grip_assist=_grasp_offsets("L", 0.83, 0.47, 0.46, -0.51, 0.28, 0.44, 0.38, 0.05, 0.64)
    + _grasp_offsets("R", 0.69, 0.15, 0.67, -0.28, 0.30, -0.37, 0.72, 0.41, 0.67) if HAND_SHAPE == "capsule" else
    _grasp_offsets("L", 0.29, 0.17, 0.93, 0.46, 0.09, -0.02, 0.49, 0.23, 0.16)
    + _grasp_offsets("R", 0.27, 0.33, 0.20, -0.07, -0.05, -0.26, 0.04, 0.00, 0.98),
)


@cache
def _compiled(name: str) -> mujoco.MjModel:
    r = ROBOTS[name]
    model = r.spec_fn().compile()
    names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt))
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE, "first joint must be the free joint"
    assert names[1:] == r.joint_names, f"joint order mismatch: {names[1:]}"
    assert model.nq == r.nq and model.nv == r.nv
    return model


def get_robot(name: str | None = None) -> RobotSpec:
    name = name or os.environ.get("SM_ROBOT", "g1")
    if name not in ROBOTS:
        raise ValueError(f"unknown robot {name!r}; choose from {sorted(ROBOTS)}")
    return ROBOTS[name]


ROBOT = get_robot()

"""Add the prop reference (broom, dustpan), the table and the glove-loaded fingers to a 50 Hz SuperMimic clip.

  SM_ROBOT=g1_inspire_sweep uv run python scripts/data/add_props_to_clip.py \
      --retarget ~/g1-sweep-retarget/out/sweep_g1_inspire_closed.npz \
      --clip data/npz/sweep_inspire_closed/sweep_task023_g1_inspire_closed.npz \
      --out data/npz/sweep_inspire_props/sweep_task023_g1_inspire_props.npz

The robot reached the clip through ``tools/export_supermimic.py`` (retarget times -> 30 Hz CSV) and
``csv_to_npz`` (30 -> 50 Hz); the prop poses go through the same two resamplings so the hand-prop relations
the retarget solved for survive. Added keys:

  prop_names (P,), prop_source_names (P,)
  prop_pos_w (T,P,3), prop_quat_w (T,P,4) wxyz, prop_lin_vel_w (T,P,3), prop_ang_vel_w (T,P,3) world frame
  table_center (3,), table_half (3,) in meters, scene_scale ()
  hand_loaded (T,H) bool for ``ROBOT.hand_joint_names``: the finger's glove regions are loaded

Fingers in ``ignore`` are never marked loaded, so they are not treated as gripping fingers: the pinky by default,
which the Inspire hand cannot bring onto the broom handle (left) and reaches the dustpan on only about half of
its loaded frames (right), see scripts/data/close_fingers_on_contact.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tyro
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_prop_hold import DRIVER_REGIONS
from close_fingers_on_contact import RETARGET_ROOT, glove_region_loaded

PROP_SOURCES = {"broom": "cleaningtools_broom", "dustpan": "cleaningtools_dustpan"}


def to_export_grid(times: np.ndarray, pos: np.ndarray, quat: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Same grid and interpolation as tools/export_supermimic.py: (pos (n,3), quat (n,4) wxyz)."""
    t = np.arange(0.0, times[-1], 1.0 / fps)
    p = np.stack([np.interp(t, times, pos[:, i]) for i in range(3)], 1)
    q = Slerp(times, Rotation.from_quat(quat, scalar_first=True))(t).as_quat(scalar_first=True)
    return p, q


def main(retarget: Path, clip: Path, out: Path, retarget_root: Path = RETARGET_ROOT, export_fps: float = 30.0,
         rest: tuple[float, float] = (0.0, 2.0), dilate_s: float = 0.04, ignore: tuple[str, ...] = ("pinky_proximal",)) -> None:
    from supermimic.data.motion_convert import quat_wxyz_to_xyzw, resample_motion
    from supermimic.robots import ROBOT

    assert ROBOT.props, "run with a robot that has props (SM_ROBOT=g1_inspire_sweep)"
    sys.path.insert(0, str(retarget_root))
    from sweep_retarget.pressure import REGIONS

    z = dict(np.load(retarget))
    c = dict(np.load(clip))
    assert bool(z.get("props_grounded", False)), "retarget predates grounded props: re-run retarget.py"
    fps = float(c["fps"][0])
    assert float(c["input_fps"][0]) == export_fps
    times = z["times"]
    T = c["joint_pos"].shape[0]

    def resample(pos: np.ndarray, quat: np.ndarray):
        p, q = to_export_grid(times, pos, quat, export_fps)
        raw = np.concatenate([p, quat_wxyz_to_xyzw(q)], 1)
        m = resample_motion(raw, export_fps, fps)
        assert m.num_frames == T, (m.num_frames, T)
        return m

    # The robot root through the same path must land on the clip's pelvis: checks the time alignment.
    root = resample(z["qpos"][:, :3], z["qpos"][:, 3:7])
    pelvis = [str(n) for n in c["body_names"]].index("pelvis")
    err = np.abs(root.root_pos - c["body_pos_w"][:, pelvis]).max()
    assert err < 1e-4, f"clip and retarget are not aligned (root differs by {err:.4f} m)"

    src = [str(n) for n in z["prop_names"]]
    pos, quat, lin, ang = [], [], [], []
    for name in ROBOT.props:
        j = src.index(PROP_SOURCES[name])
        m = resample(z["prop_pos"][:, j], z["prop_quat"][:, j])
        pos.append(m.root_pos)
        quat.append(m.root_quat)
        lin.append(m.root_lin_vel_w)
        ang.append(m.root_ang_vel_w)

    region_on = glove_region_loaded(z, retarget_root, rest)
    reg = list(REGIONS)
    t50 = np.arange(T) / fps
    k = int(round(dilate_s * fps))
    loaded = np.zeros((T, len(ROBOT.hand_joint_names)), bool)
    for i, jn in enumerate(ROBOT.hand_joint_names):
        drv = jn[2:].removesuffix("_joint")
        if drv not in DRIVER_REGIONS or drv in ignore:
            continue
        on = region_on["left" if jn[0] == "L" else "right"][:, [reg.index(r) for r in DRIVER_REGIONS[drv]]].any(1)
        on = np.interp(t50, times, on.astype(float)) > 0.5
        loaded[:, i] = np.convolve(on, np.ones(2 * k + 1), mode="same") > 0 if k else on

    scale = float(z["scale"])
    f32 = lambda a: np.stack(a, 1).astype(np.float32)  # noqa: E731
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out, **c,
        prop_names=np.array(ROBOT.props), prop_source_names=np.array([PROP_SOURCES[n] for n in ROBOT.props]),
        prop_pos_w=f32(pos), prop_quat_w=f32(quat), prop_lin_vel_w=f32(lin), prop_ang_vel_w=f32(ang),
        table_center=(z["table_center"] * scale).astype(np.float32), table_half=(z["table_half"] * scale).astype(np.float32),
        scene_scale=np.array(scale), hand_loaded=loaded, hand_loaded_joints=np.array(ROBOT.hand_joint_names),
    )
    print(f"wrote {out}: {T} frames, props {ROBOT.props}, root alignment {err * 1000:.3f} mm")
    for i, name in enumerate(ROBOT.props):
        speed = np.linalg.norm(lin[i], axis=-1)
        print(f"  {name:8s} z {pos[i][:, 2].min():.3f}..{pos[i][:, 2].max():.3f} m, speed median {np.median(speed):.2f} max {speed.max():.2f} m/s")
    print(f"  table top z {(z['table_center'][2] + z['table_half'][2]) * scale:.3f} m")
    print("  hand drivers loaded: " + ", ".join(f"{n.removesuffix('_joint')} {v:.0%}" for n, v in zip(ROBOT.hand_joint_names, loaded.mean(0))))


if __name__ == "__main__":
    tyro.cli(main)

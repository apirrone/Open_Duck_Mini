"""Run the hybrid RL/animation controller against the Open Duck Mini MuJoCo scene."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from animation.runtime import HybridController, RobotMode, WalkPolicy


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by the inverse of xyzw quaternion ``q``."""
    q_w = q[-1]
    q_vec = q[:3]
    return v * (2.0 * q_w**2 - 1.0) - np.cross(q_vec, v) * q_w * 2.0 + q_vec * np.dot(q_vec, v) * 2.0


def feet_contacts(model, data, mujoco) -> list[float]:
    contacts = [False, False]
    for index in range(data.ncon):
        contact = data.contact[index]
        first = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
        second = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
        names = {first, second}
        contacts[0] |= "foot_assembly" in names and "floor" in names
        contacts[1] |= "foot_assembly_2" in names and "floor" in names
    return [float(contact) for contact in contacts]


def observation(model, data, mujoco, commands: np.ndarray) -> np.ndarray:
    base_quat_wxyz = data.qpos[3:7].copy()
    base_quat_xyzw = np.array(
        [base_quat_wxyz[1], base_quat_wxyz[2], base_quat_wxyz[3], base_quat_wxyz[0]]
    )
    return np.concatenate(
        (
            quat_rotate_inverse(base_quat_xyzw, np.array([0.0, 0.0, -1.0])),
            data.qpos[7:23].copy(),
            data.qvel[6:22].copy(),
            feet_contacts(model, data, mujoco),
            np.zeros(16),  # WalkPolicy replaces this with its raw previous action.
            commands,
        )
    ).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx_model_path", default="BEST_WALK_ONNX_2.onnx")
    parser.add_argument("--clips", default="animation/clips")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--mode", choices=("stand", "walk", "hybrid-stand", "hybrid-walk", "dock"), default="stand")
    parser.add_argument("--play", action="append", default=[], metavar="CLIP_NAME")
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import mujoco
    except ImportError:
        print("MuJoCo is required: install with `pip install mujoco`.", file=sys.stderr)
        return 2
    try:
        import onnxruntime  # noqa: F401 - WalkPolicy gives the actionable error otherwise.
    except ImportError:
        print("onnxruntime is required: install with `pip install onnxruntime`.", file=sys.stderr)
        return 2

    root = REPOSITORY_ROOT
    model_path = Path(args.onnx_model_path)
    if not model_path.is_absolute():
        model_path = root / model_path
    if not model_path.is_file():
        print(f"ONNX model not found: {model_path}", file=sys.stderr)
        return 2
    scene_path = root / "mini_bdx/robots/open_duck_mini_v2/scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    policy = WalkPolicy(model_path)
    try:
        controller = HybridController(policy=policy, clips_dir=root / args.clips)
    except RuntimeError as exc:
        print(f"Animation runtime unavailable: {exc}", file=sys.stderr)
        return 2

    modes = {
        "stand": RobotMode.STAND,
        "walk": RobotMode.WALK,
        "hybrid-stand": RobotMode.HYBRID_STAND,
        "hybrid-walk": RobotMode.HYBRID_WALK,
        "dock": RobotMode.DEMO_DOCK,
    }
    controller.set_mode(RobotMode.STAND)
    controller.mode_machine.update(1.0)
    if modes[args.mode] is not RobotMode.STAND:
        controller.set_mode(modes[args.mode])
        controller.mode_machine.update(1.0)
    data.qpos[7:23] = controller.standing_pose
    data.ctrl[:16] = controller.standing_pose
    commands = np.array((args.vx, args.vy, args.wz), dtype=np.float32)
    for clip_name in args.play:
        if not controller.play(clip_name, commanded_velocity=commands):
            print(f"Could not start clip: {clip_name}", file=sys.stderr)
            return 2
    max_joint_deviation = 0.0
    safety_clamped = False
    max_tilt = 0.0
    control_steps = max(1, round(args.duration / 0.02))

    viewer = None
    if not args.headless:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
        print("Use --mode, --play CLIP_NAME, --vx, --vy, and --wz to configure this run.")
    try:
        for _ in range(control_steps):
            if viewer is not None and not viewer.is_running():
                break
            target = controller.step(0.02, commands, observation(model, data, mujoco, commands))
            data.ctrl[:16] = target
            max_joint_deviation = max(max_joint_deviation, float(np.max(np.abs(target - controller.standing_pose))))
            safety_clamped |= bool(controller.status()["clamped_joints"])
            for _ in range(4):
                mujoco.mj_step(model, data)
            up = quat_rotate_inverse(
                np.array([data.qpos[4], data.qpos[5], data.qpos[6], data.qpos[3]]),
                np.array([0.0, 0.0, 1.0]),
            )
            max_tilt = max(max_tilt, float(np.arccos(np.clip(up[2], -1.0, 1.0))))
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0.0, 0.02 - model.opt.timestep * 4))
    finally:
        if viewer is not None:
            viewer.close()

    upright = max_tilt < 1.0 and data.qpos[2] > 0.08
    print(
        "summary "
        f"stayed_upright={upright} "
        f"max_joint_deviation={max_joint_deviation:.4f} "
        f"safety_clamping={safety_clamped}"
    )
    return 0 if upright else 1


if __name__ == "__main__":
    raise SystemExit(main())

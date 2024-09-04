import time
import json
import os
import warnings
from placo_utils.visualization import robot_viz
import numpy as np
import placo
import FramesViewer.utils as fv_utils
from scipy.spatial.transform import Rotation as R

FPS = 60

episode = {
    "LoopMode": "Wrap",
    "FrameDuration": np.around(1 / FPS, 4),
    "EnableCycleOffsetPosition": True,
    "EnableCycleOffsetRotation": False,
    "Debug_info": [],
    "Frames": [],
    "MotionWeight": 1,
}

DT = 0.01
REFINE = 10
MESHCAT_FPS = 20
robot = placo.HumanoidRobot("../../mini_bdx/robots/bdx/robot.urdf")
solver = placo.KinematicsSolver(robot)
# viz = robot_viz(robot)

robot.set_joint_limits("left_knee", -2, -0.01)
robot.set_joint_limits("right_knee", -2, -0.01)

T_world_trunk = np.eye(4)
T_world_trunk[:3, 3] = [0, 0, 0.175]
T_world_trunk = fv_utils.rotateInSelf(T_world_trunk, [0, 2, 0])
trunk_task = solver.add_frame_task("trunk", T_world_trunk)
trunk_task.configure("trunk", "hard")

T_trunk_leftFoot = np.eye(4)
T_trunk_leftFoot[:3, 3] = [-0.03, 0.06, -0.17]
T_world_leftFoot = T_world_trunk @ T_trunk_leftFoot
T_world_leftFoot = placo.flatten_on_floor(T_world_leftFoot)
left_foot_task = solver.add_frame_task("left_foot_frame", T_world_leftFoot)
left_foot_task.configure("left_foot_frame", "soft", 1.0)

T_trunk_rightFoot = np.eye(4)
T_trunk_rightFoot[:3, 3] = [-0.03, -0.06, -0.17]
T_world_rightFoot = T_world_trunk @ T_trunk_rightFoot
T_world_rightFoot = placo.flatten_on_floor(T_world_rightFoot)
right_foot_task = solver.add_frame_task("right_foot_frame", T_world_rightFoot)
right_foot_task.configure("right_foot_frame", "soft", 1.0)


left_foot_task.orientation().mask.set_axises("yz", "local")
right_foot_task.orientation().mask.set_axises("yz", "local")


def get_angles():
    angles = {
        "left_hip_yaw": robot.get_joint("left_hip_yaw"),
        "left_hip_roll": robot.get_joint("left_hip_roll"),
        "left_hip_pitch": robot.get_joint("left_hip_pitch"),
        "left_knee": robot.get_joint("left_knee"),
        "left_ankle": robot.get_joint("left_ankle"),
        "neck_pitch": robot.get_joint("neck_pitch"),
        "head_pitch": robot.get_joint("head_pitch"),
        "head_yaw": robot.get_joint("head_yaw"),
        "left_antenna": robot.get_joint("left_antenna"),
        "right_antenna": robot.get_joint("right_antenna"),
        "right_hip_yaw": robot.get_joint("right_hip_yaw"),
        "right_hip_roll": robot.get_joint("right_hip_roll"),
        "right_hip_pitch": robot.get_joint("right_hip_pitch"),
        "right_knee": robot.get_joint("right_knee"),
        "right_ankle": robot.get_joint("right_ankle"),
    }

    return angles


last_meshcat_display = 0
last_record = 0
prev_root_position = [0, 0, 0]
prev_root_orientation_euler = [0, 0, 0]
prev_left_toe_pos = [0, 0, 0]
prev_right_toe_pos = [0, 0, 0]
prev_joints_positions = [0] * 15
prev_initialized = False

t = 0
start = 0
while True:
    # if t - last_meshcat_display > 1 / MESHCAT_FPS:
    #     last_meshcat_display = t
    #     viz.display(robot.state.q)

    trunk_task.T_world_frame = fv_utils.translateInSelf(
        T_world_trunk, [0, 0.01 * np.sin(2 * t), 0.01 * np.sin(3 * t)]
    )

    if t - last_record >= 1 / FPS:
        T_world_fbase = robot.get_T_world_fbase()
        root_position = list(T_world_fbase[:3, 3])
        root_orientation_quat = list(R.from_matrix(T_world_fbase[:3, :3]).as_quat())
        joints_positions = list(get_angles().values())

        T_world_leftFoot = robot.get_T_world_left()
        T_world_rightFoot = robot.get_T_world_right()

        T_body_leftFoot = np.linalg.inv(T_world_fbase) @ T_world_leftFoot
        T_body_rightFoot = np.linalg.inv(T_world_fbase) @ T_world_rightFoot

        left_toe_pos = list(T_body_leftFoot[:3, 3])
        right_toe_pos = list(T_body_rightFoot[:3, 3])

        world_linear_vel = list(
            (np.array(root_position) - np.array(prev_root_position)) / (1 / FPS)
        )

        world_angular_vel = list(
            (
                R.from_quat(root_orientation_quat).as_euler("xyz")
                - prev_root_orientation_euler
            )
            / (1 / FPS)
        )

        joints_vel = list(
            (np.array(joints_positions) - np.array(prev_joints_positions)) / (1 / FPS)
        )
        left_toe_vel = list(
            (np.array(left_toe_pos) - np.array(prev_left_toe_pos)) / (1 / FPS)
        )
        right_toe_vel = list(
            (np.array(right_toe_pos) - np.array(prev_right_toe_pos)) / (1 / FPS)
        )

        if prev_initialized:
            episode["Frames"].append(
                root_position
                + root_orientation_quat
                + joints_positions
                + left_toe_pos
                + right_toe_pos
                + world_linear_vel
                + world_angular_vel
                + joints_vel
                + left_toe_vel
                + right_toe_vel
            )

            episode["Debug_info"].append(
                {
                    "left_foot_pose": list(T_world_leftFoot.flatten()),
                    "right_foot_pose": list(T_world_rightFoot.flatten()),
                }
            )

        prev_root_position = root_position.copy()
        prev_root_orientation_euler = (
            R.from_quat(root_orientation_quat).as_euler("xyz").copy()
        )
        prev_left_toe_pos = left_toe_pos.copy()
        prev_right_toe_pos = right_toe_pos.copy()
        prev_joints_positions = joints_positions.copy()
        prev_initialized = True
        last_record = t
        if t - start >= 10:
            break

    robot.update_kinematics()
    _ = solver.solve(True)
    t += DT

print("recorded", len(episode["Frames"]), "frames")
file_path = "wiggle" + str(".txt")
print("DONE, saving", file_path)
with open(file_path, "w") as f:
    json.dump(episode, f)

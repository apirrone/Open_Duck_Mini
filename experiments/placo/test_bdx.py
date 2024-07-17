import argparse
import time
import warnings

import numpy as np
import placo
from placo_utils.visualization import robot_viz
from placo_utils.tf import tf
import time
import pickle
import FramesViewer.utils as fv_utils

warnings.filterwarnings("ignore")
model_filename = "../../mini_bdx/robots/bdx/robot.urdf"
robot = placo.RobotWrapper(model_filename)
robot.set_joint_limits("left_knee", -2, -0.01)
robot.set_joint_limits("right_knee", -2, -0.01)
solver = placo.KinematicsSolver(robot)
# solver.mask_fbase(True)

left_foot_task = solver.add_frame_task("left_foot_frame", np.eye(4))
left_foot_task.configure("left_foot_frame", "soft", 1.0)
right_foot_task = solver.add_frame_task("right_foot_frame", np.eye(4))
right_foot_task.configure("right_foot_frame", "soft", 1.0)

T_world_trunk = np.eye(4)
T_world_trunk = fv_utils.rotateInSelf(T_world_trunk, [0, -2, 0])
trunk_task = solver.add_frame_task("trunk", T_world_trunk)
trunk_task.configure("trunk", "hard")

robot.update_kinematics()

move = []
viz = robot_viz(robot)
while True:  # some main loop
    # Update tasks data here

    # left_foot_task.T_world_frame = tf.translation_matrix(
    #     [0, -0.12 / 2, np.sin(time.time())]
    # )
    z_offset = 0.015 * np.sin(2 * time.time())
    left_foot_task.T_world_frame = tf.translation_matrix(
        [-0.005, 0.12 / 2, -0.17 + z_offset]
    )
    right_foot_task.T_world_frame = tf.translation_matrix(
        [-0.005, -0.12 / 2, -0.17 + z_offset]
    )

    # Solve the IK
    solver.solve(True)

    # Update frames and jacobians
    robot.update_kinematics()

    angles = {
        "right_hip_yaw": robot.get_joint("right_hip_yaw"),
        "right_hip_roll": robot.get_joint("right_hip_roll"),
        "right_hip_pitch": robot.get_joint("right_hip_pitch"),
        "right_knee": robot.get_joint("right_knee"),
        "right_ankle": robot.get_joint("right_ankle"),
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
    }
    move.append(angles)
    time.sleep(1 / 60)
    pickle.dump(move, open("move.pkl", "wb"))
    # Optionally: dump the solver status
    solver.dump_status()
    viz.display(robot.state.q)

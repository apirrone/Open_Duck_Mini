import argparse
import pickle
import time

import FramesViewer.utils as fv_utils
import numpy as np

data = [1, 2, 3]
pickle.dump(data, open("data.pkl", "wb"))
from mini_bdx.hwi import HWI
from mini_bdx.utils import PolySpline
from placo_utils.visualization import robot_viz

import placo

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--real_robot", action="store_true", default=False)
args = parser.parse_args()

robot = placo.RobotWrapper(
    "../mini_bdx/robots/bdx/robot.urdf", placo.Flags.ignore_collisions
)


kinematics_solver = placo.KinematicsSolver(robot)

trunk_pitch = 0
T_world_trunk = np.eye(4)
T_world_trunk = fv_utils.rotateInSelf(T_world_trunk, [0, trunk_pitch, 0], degrees=True)
T_world_trunk = fv_utils.translateInSelf(T_world_trunk, [0.05, 0, -0.02])

T_world_head = robot.get_T_world_frame("head")
T_world_head = fv_utils.translateInSelf(T_world_head, [-0.05, 0, -0.05])

head_task = kinematics_solver.add_frame_task("head", T_world_head)
head_task.configure("head", "soft")

trunk_task = kinematics_solver.add_frame_task("trunk", T_world_trunk)
trunk_task.configure("trunk", "hard")

right_foot_task = kinematics_solver.add_frame_task(
    "right_foot", robot.get_T_world_frame("right_foot")
)
right_foot_task.configure("right_foot", "soft", 5.0, 0.1)

left_foot_task = kinematics_solver.add_frame_task(
    "left_foot", robot.get_T_world_frame("left_foot")
)
left_foot_task.configure("left_foot", "soft", 5.0, 0.1)

if not args.real_robot:
    viz = robot_viz(robot)
else:
    hwi = HWI("/dev/ttyUSB0")
    hwi.turn_on()

feet_starting_z = robot.get_T_world_frame("right_foot")[2, 3]
feet_z_spline = PolySpline()
feet_z_spline.add_point(0, feet_starting_z, 0)
feet_z_spline.add_point(3, feet_starting_z + 0.04, 0)
feet_z_spline.add_point(4, feet_starting_z + 0.04, 0)
feet_z_spline.add_point(4.1, feet_starting_z - 0.02, 0)
feet_z_spline.add_point(4.2, feet_starting_z, 0)

while True:
    t = time.time()
    tmp = left_foot_task.T_world_frame.copy()
    tmp[2, 3] = feet_z_spline.get(t % 4.5)
    left_foot_task.T_world_frame = tmp

    tmp = right_foot_task.T_world_frame.copy()
    tmp[2, 3] = feet_z_spline.get(t % 4.5)
    right_foot_task.T_world_frame = tmp

    if not args.real_robot:
        viz.display(robot.state.q)
    else:
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
        }
        hwi.set_position_all(angles)

    robot.update_kinematics()
    kinematics_solver.solve(True)

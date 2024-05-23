import time

import numpy as np
import placo
from ischedule import run_loop, schedule
from placo_utils.tf import tf
from placo_utils.visualization import robot_frame_viz, robot_viz

from mini_bdx.walk_engine import WalkEngine

robot = placo.RobotWrapper("../../robots/bdx/robot.urdf", placo.Flags.ignore_collisions)
solver = placo.KinematicsSolver(robot)
walk_engine = WalkEngine(
    robot,
    solver,
    step_size_x=0.08,
    step_size_y=0.0,
    swing_gain=0.04,
    # step_size_yaw=np.deg2rad(10),
    rise_gain=0.02,
    foot_y_offset=0.0,
    trunk_pitch=-20,
)
viz = robot_viz(robot)

t = 0
walk_engine.new_step()
start = time.time()
while True:
    t = time.time() - start

    # walk_engine.step_duration
    if t > walk_engine.step_duration:
        t = 0
        start = time.time()
        walk_engine.new_step()

    walk_engine.update(t)
    angles = walk_engine.compute_angles()
    robot.update_kinematics()
    solver.solve(True)
    # Showing effector frame
    robot_frame_viz(robot, "left_foot_tip")
    robot_frame_viz(robot, "right_foot_tip")

    # Updating the viewer
    viz.display(robot.state.q)

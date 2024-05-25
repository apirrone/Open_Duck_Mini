import time

import cv2
import numpy as np
import placo
from ischedule import run_loop, schedule
from placo_utils.tf import tf
from placo_utils.visualization import robot_frame_viz, robot_viz

from mini_bdx.walk_engine import WalkEngine

robot = placo.RobotWrapper(
    "../../mini_bdx/robots/bdx/robot.urdf", placo.Flags.ignore_collisions
)
solver = placo.KinematicsSolver(robot)
walk_engine = WalkEngine(
    robot,
    solver,
    step_size_x=0.0,
    step_size_y=0.0,
    swing_gain=0.0,
    # step_size_yaw=np.deg2rad(10),
    rise_gain=0.02,
    foot_y_offset=0.0,
    trunk_pitch=0,
    frequency=1,
)
viz = robot_viz(robot)

im = np.zeros((100, 100, 3), np.uint8)

t = 0
walk_engine.new_step()
start = time.time()
while True:
    t = time.time() - start
    target_step_size_x = 0
    target_step_size_y = 0
    target_yaw = 0
    cv2.imshow("im", im)
    key = cv2.waitKey(1)
    if key == ord("z"):
        target_step_size_x = 0.03
    if key == ord("s"):
        target_step_size_x = -0.03
    if key == ord("q"):
        target_step_size_y = 0.03
    if key == ord("d"):
        target_step_size_y = -0.03
    if key == ord("a"):
        target_yaw = np.deg2rad(10)
    if key == ord("e"):
        target_yaw = -np.deg2rad(10)
    if key == ord("o"):
        walk_engine.swing_gain -= 0.001
    if key == ord("p"):
        walk_engine.swing_gain += 0.001
    if key == ord("l"):
        walk_engine.frequency -= 0.1
    if key == ord("m"):
        walk_engine.frequency += 0.1
    print("swing gain", walk_engine.swing_gain)
    print("frequency", walk_engine.frequency)
    print("--")
    # if key == ord("n"):
    #     t += 0.01
    #     print(t)

    # walk_engine.step_duration
    if t > walk_engine.step_duration:
        t = 0
        start = time.time()
        walk_engine.new_step()

    walk_engine.update(
        True, [0, 0, 0], target_step_size_x, target_step_size_y, target_yaw, 0, 0, 0, t
    )
    angles = walk_engine.compute_angles()
    robot.update_kinematics()
    solver.solve(True)
    # Showing effector frame
    robot_frame_viz(robot, "left_foot_tip")
    robot_frame_viz(robot, "right_foot_tip")
    robot_frame_viz(robot, "head")

    # Updating the viewer
    viz.display(robot.state.q)

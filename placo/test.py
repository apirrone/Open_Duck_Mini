import numpy as np
from ischedule import run_loop, schedule
from placo_utils.tf import tf
from placo_utils.visualization import robot_frame_viz, robot_viz

import placo

robot = placo.RobotWrapper("../robots/test_bd1_frames/robot.urdf", placo.Flags.ignore_collisions)
solver = placo.KinematicsSolver(robot)
viz = robot_viz(robot)

T_world_leftFootTip = robot.get_T_world_frame("left_foot_tip")
T_world_rightFootTip = robot.get_T_world_frame("right_foot_tip")
T_world_trunk = robot.get_T_world_frame("trunk")

trunk_orientation_task = solver.add_orientation_task("trunk", np.eye(3))
trunk_orientation_task.configure("trunk", "soft")

# solver.mask_fbase(True)
# Adding a frame task
right_foot_tip_task = solver.add_frame_task("right_foot_tip", T_world_rightFootTip)
right_foot_tip_task.configure("right_foot_tip", "soft")

left_foot_tip_task = solver.add_frame_task("left_foot_tip", T_world_leftFootTip)
left_foot_tip_task.configure("left_foot_tip", "hard")

# trunk_task = solver.add_frame_task("trunk", T_world_trunk)
# trunk_task.configure("trunk", "soft")

t = 0
dt = 0.01


@schedule(interval=dt)
def loop():
    global t
    t += dt

    right_foot_tip_task.T_world_frame = T_world_rightFootTip @ tf.translation_matrix(
        [np.sin(t*3)*0.1, 0.0, 0.05]
    )

    # Updating kinematics
    robot.update_kinematics()
    solver.solve(True)

    # Showing effector frame
    robot_frame_viz(robot, "right_foot_tip")
    robot_frame_viz(robot, "left_foot_tip")

    # Updating the viewer
    viz.display(robot.state.q)


run_loop()

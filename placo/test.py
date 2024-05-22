import numpy as np
from ischedule import run_loop, schedule
from placo_utils.tf import tf
from placo_utils.visualization import robot_frame_viz, robot_viz

import placo

robot = placo.RobotWrapper("../robots/test_bd1_frames/robot.urdf")
solver = placo.KinematicsSolver(robot)
viz = robot_viz(robot)

T_world_leftFootTip = robot.get_T_world_frame("left_foot_tip").copy()
T_world_trunk = robot.get_T_world_frame("trunk")


# solver.mask_fbase(True)
# Adding a frame task
right_foot_tip_task = solver.add_frame_task("right_foot_tip", np.eye(4))
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

    right_foot_tip_task.T_world_frame = tf.translation_matrix(
        [1.25, np.sin(10 * t), 1.0]
    )

    # Updating kinematics
    robot.update_kinematics()
    solver.solve(True)

    # Showing effector frame
    robot_frame_viz(robot, "right_foot_tip")

    # Updating the viewer
    viz.display(robot.state.q)


run_loop()

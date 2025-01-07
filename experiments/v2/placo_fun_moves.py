import time
import placo
import numpy as np
from ischedule import schedule, run_loop
from placo_utils.visualization import robot_viz, robot_frame_viz, frame_viz
from placo_utils.tf import tf

from mini_bdx_runtime.hwi_feetech_pypot import HWI

MESHCAT_VIZ = False

joints = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
    # "neck_pitch",
    # "head_pitch",
    # "head_yaw",
]

if not MESHCAT_VIZ:
    hwi = HWI()
    hwi.turn_on()

time.sleep(1)
exit()


DT = 0.01
# robot = placo.HumanoidRobot("/home/antoine/MISC/mini_BDX/mini_bdx/robots/open_duck_mini_v2/robot.urdf")
robot = placo.RobotWrapper(
    "/home/antoine/MISC/mini_BDX/mini_bdx/robots/open_duck_mini_v2/robot.urdf",
    placo.Flags.ignore_collisions,
)

# Placing the left foot in world origin
robot.set_joint("left_knee", 0.1)
robot.set_joint("right_knee", 0.1)
robot.update_kinematics()
robot.set_T_world_frame("left_foot", np.eye(4))
robot.update_kinematics()

solver = placo.KinematicsSolver(robot)

# Retrieving initial position of the feet, com and trunk orientation
T_world_left = robot.get_T_world_frame("left_foot")
T_world_right = robot.get_T_world_frame("right_foot")

if MESHCAT_VIZ:
    viz = robot_viz(robot)

T_world_trunk = robot.get_T_world_frame("trunk")
T_world_trunk[2, 3] = 0.25
trunk_task = solver.add_frame_task("trunk", T_world_trunk)
trunk_task.configure("trunk_task", "soft", 1e3, 1e3)

# Keep left and right foot on the floor
left_foot_task = solver.add_frame_task("left_foot", T_world_left)
left_foot_task.configure("left_foot", "soft", 1.0, 1.0)

right_foot_task = solver.add_frame_task("right_foot", T_world_right)
right_foot_task.configure("right_foot", "soft", 1e3, 1e3)

# Regularization task
posture_regularization_task = solver.add_joints_task()
posture_regularization_task.set_joints({dof: 0.0 for dof in robot.joint_names()})
posture_regularization_task.configure("reg", "soft", 1e-5)


# Initializing robot position before enabling constraints
for _ in range(32):
    solver.solve(True)
    robot.update_kinematics()

# Enabling joint and velocity limits
solver.enable_joint_limits(True)
solver.enable_velocity_limits(True)

t = 0
dt = 0.01
last = 0
solver.dt = dt
start_t = time.time()
robot.update_kinematics()


def get_angles():
    angles = {joint: robot.get_joint(joint) for joint in joints}
    return angles


# original_T_world_frame = T_world_trunk.copy()
while True:

    # T_world_frame = original_T_world_frame.copy()

    trunk_task.T_world_frame = tf.translation_matrix([0, 0, 0.25]) @ tf.rotation_matrix(
        0.25 * np.sin(2 * np.pi * 0.5 * t), [0, 0, 1]
    )

    # y_targ = 0.05 * np.sin(2 * np.pi * 0.5 * t)
    # x_targ = 0.05 * np.cos(2 * np.pi * 0.5 * t)

    # T_world_frame[:3, 3] += [0, y_targ, 0]

    # trunk_task.T_world_frame = T_world_frame.copy()

    solver.solve(True)
    robot.update_kinematics()

    if not MESHCAT_VIZ:
        all_angles = list(get_angles().values())

        angles = {}
        for i, motor_name in enumerate(hwi.joints.keys()):
            angles[motor_name] = all_angles[i]

        hwi.set_position_all(angles)

    if MESHCAT_VIZ:
        viz.display(robot.state.q)
        robot_frame_viz(robot, "left_foot")
        frame_viz("left_foot_target", left_foot_task.T_world_frame, opacity=0.25)

    time.sleep(DT)
    t += DT
    print(t)

# @schedule(interval=dt)
# def loop():
#     global t

#     # # Updating left foot target
#     # left_foot_task.T_world_frame = tf.translation_matrix(
#     #     [np.sin(t * 2.5) * 0.05, np.sin(t * 3) * 0.1, 0.04]
#     # ) @ tf.rotation_matrix(np.sin(t) * 0.25, [1, 0, 0])

#     trunk_task.T_world_frame = tf.translation_matrix(
#         [0, 0, 0.25]
#     ) @ tf.rotation_matrix(np.sin(t) * 0.25, [1, 0, 0])

#     solver.solve(True)
#     robot.update_kinematics()


#     if MESHCAT_VIZ:
#         viz.display(robot.state.q)
#         robot_frame_viz(robot, "left_foot")
#         frame_viz("left_foot_target", left_foot_task.T_world_frame, opacity=0.25)

#     t += dt


# run_loop()
# # task =

# # while True:

# #     robot.update_kinematics()
# #     time.sleep(DT)

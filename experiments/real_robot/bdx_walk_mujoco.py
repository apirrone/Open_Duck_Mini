import argparse
import time
import warnings

import mujoco
import mujoco.viewer
import numpy as np
import placo

from mini_bdx.hwi import HWI
from mini_bdx.utils.mujoco_utils import check_contact

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--robot", action="store_true", default=False)
args = parser.parse_args()

if args.robot:
    hwi = HWI(usb_port="/dev/ttyUSB0")
    hwi.turn_on()
    time.sleep(2)

DT = 0.01
REFINE = 10
model_filename = "../../mini_bdx/robots/bdx/robot.urdf"

# Loading the robot
robot = placo.HumanoidRobot(model_filename)

# Walk parameters - if double_support_ratio is not set to 0, should be greater than replan_frequency
parameters = placo.HumanoidParameters()

parameters.double_support_ratio = 0.2  # Ratio of double support (0.0 to 1.0)
parameters.startend_double_support_ratio = (
    1.5  # Ratio duration of supports for starting and stopping walk
)
parameters.planned_timesteps = 48  # Number of timesteps planned ahead
parameters.replan_timesteps = 10  # Replanning each n timesteps
# parameters.zmp_reference_weight = 1e-6

# Posture parameters
if not args.robot:
    parameters.walk_com_height = 0.15  # Constant height for the CoM [m]
    parameters.walk_foot_height = 0.01  # Height of foot rising while walking [m]
    parameters.walk_trunk_pitch = np.deg2rad(10)  # Trunk pitch angle [rad]
    parameters.single_support_duration = 0.2  # Duration of single support phase [s]
else:
    parameters.walk_com_height = 0.17  # Constant height for the CoM [m]
    parameters.walk_foot_height = 0.02  # Height of foot rising while walking [m]
    parameters.walk_trunk_pitch = np.deg2rad(-15)  # Trunk pitch angle [rad]
    parameters.single_support_duration = 0.25  # Duration of single support phase [s]
parameters.walk_foot_rise_ratio = (
    0.2  # Time ratio for the foot swing plateau (0.0 to 1.0)
)

parameters.single_support_timesteps = (
    10  # Number of planning timesteps per single support phase
)

# Feet parameters
parameters.foot_length = 0.06  # Foot length [m]
parameters.foot_width = 0.006  # Foot width [m]
parameters.feet_spacing = 0.12  # Lateral feet spacing [m]
parameters.zmp_margin = 0.0  # ZMP margin [m]
parameters.foot_zmp_target_x = 0.0  # Reference target ZMP position in the foot [m]
parameters.foot_zmp_target_y = 0.0  # Reference target ZMP position in the foot [m]

# Limit parameters
parameters.walk_max_dtheta = 1  # Maximum dtheta per step [rad]
parameters.walk_max_dy = 0.1  # Maximum dy per step [m]
parameters.walk_max_dx_forward = 0.08  # Maximum dx per step forward [m]
parameters.walk_max_dx_backward = 0.03  # Maximum dx per step backward [m]

# Creating the kinematics solver
solver = placo.KinematicsSolver(robot)
solver.enable_velocity_limits(True)
robot.set_velocity_limits(12.0)
solver.dt = DT / REFINE

# Creating the walk QP tasks
tasks = placo.WalkTasks()
# tasks.trunk_mode = True
# tasks.com_x = 0.04
tasks.initialize_tasks(solver, robot)
tasks.left_foot_task.orientation().mask.set_axises("yz", "local")
tasks.right_foot_task.orientation().mask.set_axises("yz", "local")
# tasks.trunk_orientation_task.configure("trunk_orientation", "soft", 1e-4)
# tasks.left_foot_task.orientation().configure("left_foot_orientation", "soft", 1e-6)
# tasks.right_foot_task.orientation().configure("right_foot_orientation", "soft", 1e-6)

# # Creating a joint task to assign DoF values for upper body
joints_task = solver.add_joints_task()
joints_task.set_joints(
    {
        "head_pitch": np.deg2rad(-10),
        "head_yaw": 0.0,
        "neck_pitch": np.deg2rad(-10),
    }
)
joints_task.configure("joints", "soft", 1.0)

# cam = solver.add_centroidal_momentum_task(np.array([0., 0., 0.]))
# cam.mask.set_axises("x", "custom")
# cam.mask.R_custom_world = robot.get_T_world_frame("trunk")[:3, :3].T
# cam.configure("cam", "soft", 1e-3)

# Placing the robot in the initial position
print("Placing the robot in the initial position...")
tasks.reach_initial_pose(
    np.eye(4),
    parameters.feet_spacing,
    parameters.walk_com_height,
    parameters.walk_trunk_pitch,
)
print("Initial position reached")


# Creating the FootstepsPlanner
repetitive_footsteps_planner = placo.FootstepsPlannerRepetitive(parameters)
d_x = 0.0
d_y = 0.0
d_theta = 0.0
nb_steps = 5
repetitive_footsteps_planner.configure(d_x, d_y, d_theta, nb_steps)

# Planning footsteps
T_world_left = placo.flatten_on_floor(robot.get_T_world_left())
T_world_right = placo.flatten_on_floor(robot.get_T_world_right())
footsteps = repetitive_footsteps_planner.plan(
    placo.HumanoidRobot_Side.left, T_world_left, T_world_right
)

supports = placo.FootstepsPlanner.make_supports(
    footsteps, True, parameters.has_double_support(), True
)

# Creating the pattern generator and making an initial plan
walk = placo.WalkPatternGenerator(robot, parameters)
trajectory = walk.plan(supports, robot.com_world(), 0.0)

time_since_last_right_contact = 0.0
time_since_last_left_contact = 0.0
if not args.robot:
    model = mujoco.MjModel.from_xml_path("../../mini_bdx/robots/bdx/scene.xml")
    data = mujoco.MjData(model)
    viewer = mujoco.viewer.launch_passive(
        model,
        data,  # key_callback=key_callback
    )


def get_feet_contact():
    right_contact = check_contact(data, model, "foot_module", "floor")
    left_contact = check_contact(data, model, "foot_module_2", "floor")
    return right_contact, left_contact


# Timestamps
start_t = time.time()
initial_delay = -3.0
t = initial_delay
last_display = time.time()
last_replan = 0
petage_de_gueule = False
while True:

    # Invoking the IK QP solver
    for k in range(REFINE):
        # Updating the QP tasks from planned trajectory
        if not petage_de_gueule:
            tasks.update_tasks_from_trajectory(trajectory, t - DT + k * DT / REFINE)

        robot.update_kinematics()
        qd_sol = solver.solve(True)
    # solver.dump_status()

    # Ensuring the robot is kinematically placed on the floor on the proper foot to avoid integration drifts
    # if not trajectory.support_is_both(t):
    # robot.update_support_side(str(trajectory.support_side(t)))
    # robot.ensure_on_floor()

    # If enough time elapsed and we can replan, do the replanning
    if (
        t - last_replan > parameters.replan_timesteps * parameters.dt()
        and walk.can_replan_supports(trajectory, t)
    ):
        last_replan = t

        # Replanning footsteps from current trajectory
        supports = walk.replan_supports(repetitive_footsteps_planner, trajectory, t)

        # Replanning CoM trajectory, yielding a new trajectory we can switch to
        trajectory = walk.replan(supports, trajectory, t)

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

    if not args.robot:
        right_contact, left_contact = get_feet_contact()
        if left_contact:
            time_since_last_left_contact = 0.0
        if right_contact:
            time_since_last_right_contact = 0.0

        data.ctrl[:] = list(angles.values())

        if (
            time_since_last_left_contact > parameters.single_support_duration
            or time_since_last_right_contact > parameters.single_support_duration
        ):
            petage_de_gueule = True
        else:
            petage_de_gueule = False

        time_since_last_left_contact += DT
        time_since_last_right_contact += DT

        mujoco.mj_step(model, data, 4)  # 4 seems good
        viewer.sync()
    else:
        hwi.set_position_all(angles)

    t += DT
    while time.time() < start_t + t:
        time.sleep(1e-3)

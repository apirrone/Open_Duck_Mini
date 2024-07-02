import time
import warnings

import numpy as np
import placo

warnings.filterwarnings("ignore")

DT = 0.01
REFINE = 10


class PlacoWalkEngine:
    def __init__(
        self,
        model_filename: str = "../../robots/bdx/robot.urdf",
        ignore_feet_contact: bool = False,
    ) -> None:

        self.model_filename = model_filename
        self.ignore_feet_contact = ignore_feet_contact

        # Loading the robot
        self.robot = placo.HumanoidRobot(model_filename)

        # Walk parameters - if double_support_ratio is not set to 0, should be greater than replan_frequency
        self.parameters = placo.HumanoidParameters()

        self.parameters.double_support_ratio = (
            0.2  # Ratio of double support (0.0 to 1.0)
        )
        self.parameters.startend_double_support_ratio = (
            1.5  # Ratio duration of supports for starting and stopping walk
        )
        self.parameters.planned_timesteps = 48  # Number of timesteps planned ahead
        self.parameters.replan_timesteps = 10  # Replanning each n timesteps
        # parameters.zmp_reference_weight = 1e-6

        # Posture parameters
        self.parameters.walk_com_height = 0.15  # Constant height for the CoM [m]
        self.parameters.walk_foot_height = (
            0.025  # Height of foot rising while walking [m]
        )
        self.parameters.walk_trunk_pitch = np.deg2rad(10)  # Trunk pitch angle [rad]
        self.parameters.walk_foot_rise_ratio = (
            0.2  # Time ratio for the foot swing plateau (0.0 to 1.0)
        )
        self.parameters.single_support_duration = (
            0.18  # Duration of single support phase [s]
        )
        self.parameters.single_support_timesteps = (
            10  # Number of planning timesteps per single support phase
        )

        # Feet parameters
        self.parameters.foot_length = 0.06  # Foot length [m]
        self.parameters.foot_width = 0.006  # Foot width [m]
        self.parameters.feet_spacing = 0.12  # Lateral feet spacing [m]
        self.parameters.zmp_margin = 0.0  # ZMP margin [m]
        self.parameters.foot_zmp_target_x = (
            0.0  # Reference target ZMP position in the foot [m]
        )
        self.parameters.foot_zmp_target_y = (
            0.0  # Reference target ZMP position in the foot [m]
        )

        # Limit parameters
        self.parameters.walk_max_dtheta = 1  # Maximum dtheta per step [rad]
        self.parameters.walk_max_dy = 0.1  # Maximum dy per step [m]
        self.parameters.walk_max_dx_forward = 0.08  # Maximum dx per step forward [m]
        self.parameters.walk_max_dx_backward = 0.03  # Maximum dx per step backward [m]

        # Creating the kinematics solver
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.enable_velocity_limits(True)
        self.robot.set_velocity_limits(12.0)
        self.solver.dt = DT / REFINE

        # Creating the walk QP tasks
        self.tasks = placo.WalkTasks()
        # tasks.trunk_mode = True
        # tasks.com_x = 0.04
        self.tasks.initialize_tasks(self.solver, self.robot)
        self.tasks.left_foot_task.orientation().mask.set_axises("yz", "local")
        self.tasks.right_foot_task.orientation().mask.set_axises("yz", "local")
        # tasks.trunk_orientation_task.configure("trunk_orientation", "soft", 1e-4)
        # tasks.left_foot_task.orientation().configure("left_foot_orientation", "soft", 1e-6)
        # tasks.right_foot_task.orientation().configure("right_foot_orientation", "soft", 1e-6)

        # # Creating a joint task to assign DoF values for upper body
        self.joints_task = self.solver.add_joints_task()
        self.joints_task.set_joints(
            {
                "head_pitch": np.deg2rad(-10),
                "head_yaw": 0.0,
                "neck_pitch": np.deg2rad(-10),
            }
        )
        self.joints_task.configure("joints", "soft", 1.0)

        # cam = solver.add_centroidal_momentum_task(np.array([0., 0., 0.]))
        # cam.mask.set_axises("x", "custom")
        # cam.mask.R_custom_world = robot.get_T_world_frame("trunk")[:3, :3].T
        # cam.configure("cam", "soft", 1e-3)

        # Placing the robot in the initial position
        print("Placing the robot in the initial position...")
        self.tasks.reach_initial_pose(
            np.eye(4),
            self.parameters.feet_spacing,
            self.parameters.walk_com_height,
            self.parameters.walk_trunk_pitch,
        )
        print("Initial position reached")

        # Creating the FootstepsPlanner
        self.repetitive_footsteps_planner = placo.FootstepsPlannerRepetitive(
            self.parameters
        )
        self.d_x = 0.0
        self.d_y = 0.0
        self.d_theta = 0.0
        self.nb_steps = 5
        self.repetitive_footsteps_planner.configure(
            self.d_x, self.d_y, self.d_theta, self.nb_steps
        )

        # Planning footsteps
        self.T_world_left = placo.flatten_on_floor(self.robot.get_T_world_left())
        self.T_world_right = placo.flatten_on_floor(self.robot.get_T_world_right())
        self.footsteps = self.repetitive_footsteps_planner.plan(
            placo.HumanoidRobot_Side.left, self.T_world_left, self.T_world_right
        )

        self.supports = placo.FootstepsPlanner.make_supports(
            self.footsteps, True, self.parameters.has_double_support(), True
        )

        # Creating the pattern generator and making an initial plan
        self.walk = placo.WalkPatternGenerator(self.robot, self.parameters)
        self.trajectory = self.walk.plan(self.supports, self.robot.com_world(), 0.0)

        self.time_since_last_right_contact = 0.0
        self.time_since_last_left_contact = 0.0
        self.start = None
        self.initial_delay = -3.0
        self.t = self.initial_delay
        self.last_replan = 0

    def get_angles(self):

        angles = {
            "right_hip_yaw": self.robot.get_joint("right_hip_yaw"),
            "right_hip_roll": self.robot.get_joint("right_hip_roll"),
            "right_hip_pitch": self.robot.get_joint("right_hip_pitch"),
            "right_knee": self.robot.get_joint("right_knee"),
            "right_ankle": self.robot.get_joint("right_ankle"),
            "left_hip_yaw": self.robot.get_joint("left_hip_yaw"),
            "left_hip_roll": self.robot.get_joint("left_hip_roll"),
            "left_hip_pitch": self.robot.get_joint("left_hip_pitch"),
            "left_knee": self.robot.get_joint("left_knee"),
            "left_ankle": self.robot.get_joint("left_ankle"),
            "neck_pitch": self.robot.get_joint("neck_pitch"),
            "head_pitch": self.robot.get_joint("head_pitch"),
            "head_yaw": self.robot.get_joint("head_yaw"),
            "left_antenna": self.robot.get_joint("left_antenna"),
            "right_antenna": self.robot.get_joint("right_antenna"),
        }

        return angles

    def tick(self, dt, left_contact=True, right_contact=True):
        if self.start is None:
            self.start = time.time()

        if not self.ignore_feet_contact:
            if left_contact:
                self.time_since_last_left_contact = 0.0
            if right_contact:
                self.time_since_last_right_contact = 0.0

        falling = not self.ignore_feet_contact and (
            self.time_since_last_left_contact > self.parameters.single_support_duration
            or self.time_since_last_right_contact
            > self.parameters.single_support_duration
        )

        self.repetitive_footsteps_planner.configure(
            self.d_x, self.d_y, self.d_theta, self.nb_steps
        )

        for k in range(REFINE):
            # Updating the QP tasks from planned trajectory
            if not falling:
                self.tasks.update_tasks_from_trajectory(
                    self.trajectory, self.t - dt + k * dt / REFINE
                )

            self.robot.update_kinematics()
            _ = self.solver.solve(True)

        # If enough time elapsed and we can replan, do the replanning
        if (
            self.t - self.last_replan
            > self.parameters.replan_timesteps * self.parameters.dt()
            and self.walk.can_replan_supports(self.trajectory, self.t)
        ):
            self.last_replan = self.t

            # Replanning footsteps from current trajectory
            self.supports = self.walk.replan_supports(
                self.repetitive_footsteps_planner, self.trajectory, self.t
            )

            # Replanning CoM trajectory, yielding a new trajectory we can switch to
            self.trajectory = self.walk.replan(self.supports, self.trajectory, self.t)

        self.time_since_last_left_contact += dt
        self.time_since_last_right_contact += dt
        self.t += dt

        # while time.time() < self.start_t + self.t:
        #     time.sleep(1e-3)

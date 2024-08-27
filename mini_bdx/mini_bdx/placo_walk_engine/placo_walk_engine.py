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
        self.parameters.walk_com_height = 0.175  # Constant height for the CoM [m]
        # self.parameters.walk_com_height = 0.18  # Constant height for the CoM [m]
        self.parameters.walk_foot_height = (
            0.03  # Height of foot rising while walking [m] # 3
        )
        # self.parameters.walk_trunk_pitch = 0  # Trunk pitch angle [rad]
        self.parameters.walk_trunk_pitch = np.deg2rad(5)  # Trunk pitch angle [rad]
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
        # self.parameters.foot_width = 0.006  # Foot width [m]
        self.parameters.feet_spacing = 0.14  # Lateral feet spacing [m] # 12
        self.parameters.zmp_margin = 0.00  # ZMP margin [m]
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
        self.solver.enable_joint_limits(False)
        self.solver.dt = DT / REFINE

        self.robot.set_joint_limits("left_knee", -2, -0.01)
        self.robot.set_joint_limits("right_knee", -2, -0.01)

        # Creating the walk QP tasks
        self.tasks = placo.WalkTasks()
        # tasks.trunk_mode = True
        # self.tasks.com_x = -0.015
        self.tasks.com_x = 0.0
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
                "left_antenna": np.deg2rad(0),
                "right_antenna": np.deg2rad(0),
                # "right_knee": np.deg2rad(-10),
                # "left_knee": np.deg2rad(-10),
            }
        )
        self.joints_task.configure("joints", "soft", 1.0)

        # Placing the robot in the initial position
        print("Placing the robot in the initial position...")
        self.tasks.reach_initial_pose(
            np.eye(4),
            self.parameters.feet_spacing,
            self.parameters.walk_com_height,
            self.parameters.walk_trunk_pitch,
        )
        print("Initial position reached")

        print(self.get_angles())
        # exit()

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
        self.initial_delay = -1.0
        # self.initial_delay = 0
        self.t = self.initial_delay
        self.last_replan = 0

        # TODO remove startend_double_support_duration() when starting and ending ?
        self.period = (
            2 * self.parameters.single_support_duration
            + 2 * self.parameters.double_support_duration()
        )

    def get_angles(self):
        angles = {
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
            "right_hip_yaw": self.robot.get_joint("right_hip_yaw"),
            "right_hip_roll": self.robot.get_joint("right_hip_roll"),
            "right_hip_pitch": self.robot.get_joint("right_hip_pitch"),
            "right_knee": self.robot.get_joint("right_knee"),
            "right_ankle": self.robot.get_joint("right_ankle"),
        }

        return angles

    def reset(self):
        self.t = 0
        self.start = None
        self.last_replan = 0
        self.time_since_last_right_contact = 0.0
        self.time_since_last_left_contact = 0.0

        self.tasks.reach_initial_pose(
            np.eye(4),
            self.parameters.feet_spacing,
            self.parameters.walk_com_height,
            self.parameters.walk_trunk_pitch,
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
        self.trajectory = self.walk.plan(self.supports, self.robot.com_world(), 0.0)

    def set_traj(self, d_x, d_y, d_theta):
        self.d_x = d_x
        self.d_y = d_y
        self.d_theta = d_theta
        self.repetitive_footsteps_planner.configure(
            self.d_x, self.d_y, self.d_theta, self.nb_steps
        )

    def get_footsteps_in_world(self):
        footsteps = self.trajectory.get_supports()
        footsteps_in_world = []
        for footstep in footsteps:
            if not footstep.is_both():
                footsteps_in_world.append(footstep.frame())

        for i in range(len(footsteps_in_world)):
            footsteps_in_world[i][:3, 3][1] += self.parameters.feet_spacing / 2

        return footsteps_in_world

    def get_footsteps_in_robot_frame(self):
        T_world_fbase = self.robot.get_T_world_fbase()

        footsteps = self.trajectory.get_supports()
        footsteps_in_robot_frame = []
        for footstep in footsteps:
            if not footstep.is_both():
                T_world_footstepFrame = footstep.frame().copy()
                T_fbase_footstepFrame = (
                    np.linalg.inv(T_world_fbase) @ T_world_footstepFrame
                )
                T_fbase_footstepFrame = placo.flatten_on_floor(T_fbase_footstepFrame)
                T_fbase_footstepFrame[:3, 3][2] = -T_world_fbase[:3, 3][2]

                footsteps_in_robot_frame.append(T_fbase_footstepFrame)

        return footsteps_in_robot_frame

    def get_current_support_phase(self):
        if self.trajectory.support_is_both(self.t):
            return "both"

        return self.trajectory.support_side(self.t)

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

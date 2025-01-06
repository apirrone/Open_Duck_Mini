import time
import warnings
import json

import numpy as np
import placo
import os

warnings.filterwarnings("ignore")

DT = 0.01
REFINE = 10


class PlacoWalkEngine:
    def __init__(
        self,
        asset_path: str = "",
        model_filename: str = "go_bdx.urdf",
        init_params: dict = {},
        ignore_feet_contact: bool = False,
    ) -> None:
        model_filename = os.path.join(asset_path, model_filename)
        self.asset_path = asset_path
        self.model_filename = model_filename
        self.ignore_feet_contact = ignore_feet_contact

        # Loading the robot
        self.robot = placo.HumanoidRobot(model_filename)

        self.parameters = placo.HumanoidParameters()
        if init_params is not None:
            self.load_parameters(init_params)
        else:
            defaults_filename = os.path.join(asset_path, "placo_defaults.json")
            self.load_defaults(defaults_filename)

        # Creating the kinematics solver
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.enable_velocity_limits(True)
        self.robot.set_velocity_limits(12.0)
        self.solver.enable_joint_limits(False)
        self.solver.dt = DT / REFINE

        self.robot.set_joint_limits("left_knee", 2, 0.01)
        self.robot.set_joint_limits("right_knee", 2, 0.01)

        # Creating the walk QP tasks
        self.tasks = placo.WalkTasks()
        if hasattr(self.parameters, "trunk_mode"):
            self.tasks.trunk_mode = self.parameters.trunk_mode
        self.tasks.com_x = 0.0
        self.tasks.initialize_tasks(self.solver, self.robot)
        self.tasks.left_foot_task.orientation().mask.set_axises("yz", "local")
        self.tasks.right_foot_task.orientation().mask.set_axises("yz", "local")
        # tasks.trunk_orientation_task.configure("trunk_orientation", "soft", 1e-4)
        # tasks.left_foot_task.orientation().configure("left_foot_orientation", "soft", 1e-6)
        # tasks.right_foot_task.orientation().configure("right_foot_orientation", "soft", 1e-6)

        # # Creating a joint task to assign DoF values for upper body
        self.joints = self.parameters.joints
        joint_degrees = self.parameters.joint_angles
        joint_radians = {
            joint: np.deg2rad(degrees) for joint, degrees in joint_degrees.items()
        }
        self.joints_task = self.solver.add_joints_task()
        self.joints_task.set_joints(joint_radians)
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

    def load_defaults(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        params = self.parameters
        load_parameters(data)

    def load_parameters(self, data):
        params = self.parameters
        params.double_support_ratio = data.get(
            "double_support_ratio", params.double_support_ratio
        )
        params.startend_double_support_ratio = data.get(
            "startend_double_support_ratio", params.startend_double_support_ratio
        )
        params.planned_timesteps = data.get(
            "planned_timesteps", params.planned_timesteps
        )
        params.replan_timesteps = data.get("replan_timesteps", params.replan_timesteps)
        params.walk_com_height = data.get("walk_com_height", params.walk_com_height)
        params.walk_foot_height = data.get("walk_foot_height", params.walk_foot_height)
        params.walk_trunk_pitch = np.deg2rad(
            data.get("walk_trunk_pitch", np.rad2deg(params.walk_trunk_pitch))
        )
        params.walk_foot_rise_ratio = data.get(
            "walk_foot_rise_ratio", params.walk_foot_rise_ratio
        )
        params.single_support_duration = data.get(
            "single_support_duration", params.single_support_duration
        )
        params.single_support_timesteps = data.get(
            "single_support_timesteps", params.single_support_timesteps
        )
        params.foot_length = data.get("foot_length", params.foot_length)
        params.feet_spacing = data.get("feet_spacing", params.feet_spacing)
        params.zmp_margin = data.get("zmp_margin", params.zmp_margin)
        params.foot_zmp_target_x = data.get(
            "foot_zmp_target_x", params.foot_zmp_target_x
        )
        params.foot_zmp_target_y = data.get(
            "foot_zmp_target_y", params.foot_zmp_target_y
        )
        params.walk_max_dtheta = data.get("walk_max_dtheta", params.walk_max_dtheta)
        params.walk_max_dy = data.get("walk_max_dy", params.walk_max_dy)
        params.walk_max_dx_forward = data.get(
            "walk_max_dx_forward", params.walk_max_dx_forward
        )
        params.walk_max_dx_backward = data.get(
            "walk_max_dx_backward", params.walk_max_dx_backward
        )
        params.joints = data.get("joints", [])
        params.joint_angles = data.get("joint_angles", [])
        if "trunk_mode" in data:
            params.trunk_mode = data.get("trunk_mode")

    def get_angles(self):
        angles = {joint: self.robot.get_joint(joint) for joint in self.joints}
        return angles

    def reset(self):
        self.t = self.initial_delay
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

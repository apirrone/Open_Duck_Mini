import numpy as np
import placo
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation as R

from mini_bdx.placo_walk_engine import PlacoWalkEngine
from mini_bdx.utils.mujoco_utils import get_contact_force

FRAME_SKIP = 4


class BDXEnv(MujocoEnv, utils.EzPickle):
    # Inspired by https://arxiv.org/pdf/2207.12644
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 125,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        self.nb_dofs = 15

        observation_space = Box(
            np.array(
                [
                    *(-np.pi * np.ones(self.nb_dofs)),  # joints_rotations
                    *(-10 * np.ones(self.nb_dofs)),  # joints_velocities
                    *(-10 * np.ones(3)),  # angular_velocity
                    *(-10 * np.ones(3)),  # linear_velocity
                    *(-10 * np.ones(8)),  # next two footsteps 2*[x, y, z, theta]
                    *(-np.pi * np.ones(2)),  # clock signal
                ]
            ),
            np.array(
                [
                    *(np.pi * np.ones(self.nb_dofs)),  # joints_rotations
                    *(10 * np.ones(self.nb_dofs)),  # joints_velocities
                    *(10 * np.ones(3)),  # angular_velocity
                    *(10 * np.ones(3)),  # linear_velocity
                    *(10 * np.ones(8)),  # next two footsteps 2*[x, y, z, theta]
                    *(np.pi * np.ones(2)),  # clock signal
                ]
            ),
        )

        self.prev_action = np.zeros(self.nb_dofs)
        self.prev_torque = np.zeros(self.nb_dofs)

        self.prev_t = 0
        self.init_pos = np.array(
            [
                -0.013946457213457239,
                0.07918837709879874,
                0.5325073962634973,
                -1.6225192902713386,
                0.9149246381274986,
                0.013627156377842975,
                0.07738878096596595,
                0.5933527914082196,
                -1.630548419252953,
                0.8621333440557593,
                -0.17453292519943295,
                -0.17453292519943295,
                8.65556854322817e-27,
                0,
                0,
            ]
        )

        self.pwe = PlacoWalkEngine(
            "/home/antoine/MISC/mini_BDX/mini_bdx/robots/bdx/robot.urdf",
            ignore_feet_contact=True,
        )

        self.startup_cooldown = -self.pwe.initial_delay
        self.next_footsteps = self.pwe.get_footsteps_in_robot_frame().copy()
        self.next_footsteps_world = self.pwe.get_footsteps_in_world().copy()
        for i in range(len(self.next_footsteps)):
            self.next_footsteps[i][:3, 3][2] = 0

        MujocoEnv.__init__(
            self,
            "/home/antoine/MISC/mini_BDX/mini_bdx/robots/bdx/scene.xml",
            FRAME_SKIP,
            observation_space=observation_space,
            **kwargs,
        )

    def is_terminated(self) -> bool:
        rot = np.array(self.data.body("base").xmat).reshape(3, 3)
        Z_vec = rot[:, 2]
        Z_vec /= np.linalg.norm(Z_vec)
        upright = np.array([0, 0, 1])
        return (
            self.data.body("base").xpos[2] < 0.08 or np.dot(upright, Z_vec) <= 0.4
        )  # base z is below 0.08m or base has more than 90 degrees of tilt

    def gait_reward(self):
        # During single support:
        #   - reward force on the supporting foot and speed on the flying foot
        #   - penalize force on the flying foot and speed on the supporting foot
        # During double support:
        #   - reward force on both feet
        #   - penalize speed on both feet

        support_phase = self.pwe.get_current_support_phase()  # left, right, both
        right_contact_force = np.sum(
            get_contact_force(self.data, self.model, "right_foot", "floor")
        )
        left_contact_force = np.sum(
            get_contact_force(self.data, self.model, "left_foot", "floor")
        )
        right_speed = self.data.body("right_foot").cvel[3:]  # [rot:vel] size 6
        left_speed = self.data.body("left_foot").cvel[3:]  # [rot:vel] size 6

        if support_phase == "both":
            return (
                right_contact_force
                + left_contact_force
                - np.linalg.norm(right_speed)
                - np.linalg.norm(left_speed)
            )
        elif support_phase is placo.HumanoidRobot_Side.left:
            return (
                left_contact_force
                - np.linalg.norm(left_speed)
                + right_contact_force
                + np.linalg.norm(right_speed)
            )
        elif support_phase is placo.HumanoidRobot_Side.right:
            return (
                right_contact_force
                - np.linalg.norm(right_speed)
                - left_contact_force
                + np.linalg.norm(left_speed)
            )

    def step_reward(self):
        # Incentivize the robot to step and orient the body toward targets
        # dfoot : distance of the closest foot to the upcoming footstep
        # hit reward : reward any foot that hits the upcoming footstep. Only when either or both feet are within a radius of the target
        # progress reward : encourage the moving base to move toward he target

        target_radius = (
            0.1  # Only when either or both feet are within a radius of the target ??
        )

        base_pos_2D = self.data.body("base").xpos[:2]
        upcoming_footstep = self.next_footsteps[0]
        second_footstep = self.next_footsteps[1]

        base_target_2D = np.mean(
            [upcoming_footstep[:3, 3][:2], second_footstep[:3, 3][:2]], axis=0
        )

        pos = self.data.body("base").xpos
        mat = self.data.body("base").xmat
        T_world_body = np.eye(4)
        T_world_body[:3, :3] = mat.reshape(3, 3)
        T_world_body[:3, 3] = pos

        T_world_rightFoot = np.eye(4)
        T_world_rightFoot = np.eye(4)
        pos = self.data.body("right_foot").xpos
        mat = self.data.body("right_foot").xmat
        T_world_rightFoot[:3, 3] = pos
        T_world_rightFoot[:3, :3] = mat.reshape(3, 3)

        T_world_leftFoot = np.eye(4)
        T_world_leftFoot = np.eye(4)
        pos = self.data.body("left_foot").xpos
        mat = self.data.body("left_foot").xmat
        T_world_leftFoot[:3, 3] = pos
        T_world_leftFoot[:3, :3] = mat.reshape(3, 3)

        T_body_rightFoot = np.linalg.inv(T_world_body) @ T_world_rightFoot
        T_body_leftFoot = np.linalg.inv(T_world_body) @ T_world_leftFoot

        # right_foot_pos = self.data.body("right_foot").xpos  # right
        # left_foot_pos = self.data.body("left_foot").xpos  # left

        right_foot_dist = np.linalg.norm(
            upcoming_footstep[:3, 3] - T_body_rightFoot[:3, 3]
        )
        left_foot_dist = np.linalg.norm(
            upcoming_footstep[:3, 3] - T_body_leftFoot[:3, 3]
        )

        dfoot = min(right_foot_dist, left_foot_dist)
        droot = np.linalg.norm(base_pos_2D - base_target_2D)

        khit = 0.8
        return khit * np.exp(-dfoot / 0.25) + (1 - khit) * np.exp(-droot / 2)

    def orient_reward(self):
        euler = R.from_matrix(self.pwe.robot.get_T_world_fbase()[:3, :3]).as_euler(
            "xyz"
        )
        desired_yaw = euler[2]
        current_yaw = R.from_matrix(
            np.array(self.data.body("base").xmat).reshape(3, 3)
        ).as_euler("xyz")[2]
        return (desired_yaw - current_yaw) ** 2

    def height_reward(self):
        current_height = self.data.body("base").xpos[2]
        return np.exp(-40 * (0.15 - current_height) ** 2)

    def upright_reward(self):
        # angular distance to upright position in reward
        Z_vec = np.array(self.data.body("base").xmat).reshape(3, 3)[:, 2]
        return np.square(np.dot(np.array([0, 0, 1]), Z_vec))

    def action_reward(self, a):
        current_action = a.copy()

        # This can explode, don't understand why
        return min(
            2, np.exp(-5 * np.sum((self.prev_action - current_action)) / self.nb_dofs)
        )

    def torque_reward(self):
        current_torque = self.data.qfrc_actuator
        return np.exp(
            -0.25 * np.sum((self.prev_torque - current_torque)) / self.nb_dofs
        )

    def step(self, a):

        t = self.data.time
        dt = t - self.prev_t
        if self.startup_cooldown > 0:
            self.startup_cooldown -= dt

        if self.startup_cooldown > 0:
            self.do_simulation(self.init_pos, FRAME_SKIP)
            reward = 0
        else:

            # We want to learn deltas from the initial position
            a += self.init_pos

            # Maybe use that too :)
            # current_ctrl = self.data.ctrl.copy()
            # delta_max = 0.1
            # a = np.clip(a, current_ctrl - delta_max, current_ctrl + delta_max)

            self.do_simulation(a, FRAME_SKIP)

            self.pwe.tick(dt)

            print("Gait reward: ", 0.15 * self.gait_reward())
            print("Step reward: ", 0.45 * self.step_reward())
            print("Orient reward: ", 0.05 * self.orient_reward())
            print("Height reward: ", 0.05 * self.height_reward())
            print("Upright reward: ", 0.05 * self.upright_reward())
            print("Action reward: ", 0.05 * self.action_reward(a))
            print("Torque reward: ", 0.05 * self.torque_reward())
            print("===")

            reward = (
                0.05
                + 0.15 * self.gait_reward()
                + 0.45 * self.step_reward()
                + 0.05 * self.orient_reward()
                + 0.05 * self.height_reward()
                + 0.05 * self.upright_reward()
                + 0.05 * self.action_reward(a)
                + 0.05 * self.torque_reward()
            )

        ob = self._get_obs()

        if self.render_mode == "human":
            self.render()

        self.prev_t = t
        self.prev_action = a.copy()
        self.prev_torque = self.data.qfrc_actuator.copy()

        # self.viz.display(self.pwe.robot.state.q)
        return (ob, reward, self.is_terminated(), False, {})  # terminated  # truncated

    def reset_model(self):
        self.prev_t = self.data.time
        self.startup_cooldown = 1.0
        self.prev_action = np.zeros(self.nb_dofs)
        self.prev_torque = np.zeros(self.nb_dofs)
        self.pwe.reset()

        d_x = np.random.uniform(0.01, 0.03)
        d_y = np.random.uniform(-0.01, 0.01)
        d_theta = np.random.uniform(-0.1, 0.1)
        self.pwe.set_traj(d_x, d_y, d_theta)

        self.goto_init()

        self.set_state(self.data.qpos, self.data.qvel)
        return self._get_obs()

    def goto_init(self):
        self.data.qvel[:] = np.zeros(len(self.data.qvel[:]))
        noise = np.random.uniform(-0.01, 0.01, self.nb_dofs)
        self.data.qpos[7 : 7 + self.nb_dofs] = self.init_pos + noise
        self.data.qpos[2] = 0.15
        self.data.qpos[3 : 3 + 4] = [1, 0, 0.08, 0]

        self.data.ctrl[:] = self.init_pos

    def get_clock_signal(self):
        a = np.sin(2 * np.pi * (self.data.time % self.pwe.period)) / self.pwe.period
        b = np.cos(2 * np.pi * (self.data.time % self.pwe.period)) / self.pwe.period
        return [a, b]

    def _get_obs(self):

        joints_rotations = self.data.qpos[7 : 7 + self.nb_dofs]
        joints_velocities = self.data.qvel[6 : 6 + self.nb_dofs]

        angular_velocity = self.data.body("base").cvel[
            :3
        ]  # TODO this is imu, add noise to it later
        linear_velocity = self.data.body("base").cvel[3:]

        self.next_footsteps = self.pwe.get_footsteps_in_robot_frame().copy()
        self.next_footsteps_world = self.pwe.get_footsteps_in_world().copy()
        for i in range(len(self.next_footsteps)):
            self.next_footsteps[i][:3, 3][2] = 0
        next_two_footsteps = []  # 2*[x, y, z, theta]
        for footstep in self.next_footsteps[:2]:
            yaw = R.from_matrix(footstep[:3, :3]).as_euler("xyz")[2]
            next_two_footsteps.append(list(footstep[:3, 3]) + [yaw])

        return np.concatenate(
            [
                joints_rotations,
                joints_velocities,
                angular_velocity,
                linear_velocity,
                np.array(next_two_footsteps).flatten(),
                self.get_clock_signal(),
            ]
        )

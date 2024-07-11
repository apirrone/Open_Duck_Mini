import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation as R

from mini_bdx.utils.mujoco_utils import check_contact, get_contact_force

FRAME_SKIP = 4


class BDXEnv(MujocoEnv, utils.EzPickle):
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
                    *(-10 * np.ones(3)),  # target velocity [x, y, theta]
                    *(0 * np.ones(2)),  # feet contact [left, right]
                    *(-np.pi * np.ones(2)),  # clock signal
                ]
            ),
            np.array(
                [
                    *(np.pi * np.ones(self.nb_dofs)),  # joints_rotations
                    *(10 * np.ones(self.nb_dofs)),  # joints_velocities
                    *(10 * np.ones(3)),  # angular_velocity
                    *(10 * np.ones(3)),  # linear_velocity
                    *(10 * np.ones(3)),  # target velocity [x, y, theta]
                    *(1 * np.ones(2)),  # feet contact [left, right]
                    *(np.pi * np.ones(2)),  # clock signal
                ]
            ),
        )

        self.right_foot_contact = True
        self.left_foot_contact = True

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

        self.startup_cooldown = 1.0
        self.walk_period = 1.0
        self.target_velocities = np.asarray([0, 0, 0])  # x, y, yaw
        self.cumulated_reward = 0.0
        self.last_time_both_feet_on_the_ground = 0
        self.init_pos_noise = np.zeros(self.nb_dofs)

        MujocoEnv.__init__(
            self,
            "../../../mini_bdx/robots/bdx/scene.xml",
            FRAME_SKIP,
            observation_space=observation_space,
            **kwargs,
        )

    def is_terminated(self) -> bool:
        left_antenna_contact = check_contact(
            self.data, self.model, "left_antenna_assembly", "floor"
        )
        right_antenna_contact = check_contact(
            self.data, self.model, "right_antenna_assembly", "floor"
        )
        body_contact = check_contact(self.data, self.model, "body_module", "floor")
        rot = np.array(self.data.body("base").xmat).reshape(3, 3)
        Z_vec = rot[:, 2]
        Z_vec /= np.linalg.norm(Z_vec)
        upright = np.array([0, 0, 1])

        return (
            self.data.body("base").xpos[2] < 0.08
            or np.dot(upright, Z_vec) <= 0.4
            or left_antenna_contact
            or right_antenna_contact
            or body_contact
        )

    def support_flying_reward(self):
        # Idea : reward when there is a support foot and a flying foot
        # penalize when both feet are in the air or both feet are on the ground
        right_contact_force = abs(
            np.sum(get_contact_force(self.data, self.model, "right_foot", "floor"))
        )
        left_contact_force = abs(
            np.sum(get_contact_force(self.data, self.model, "left_foot", "floor"))
        )
        right_speed = np.linalg.norm(
            self.data.body("right_foot").cvel[3:]
        )  # [rot:vel] size 6
        left_speed = np.linalg.norm(
            self.data.body("left_foot").cvel[3:]
        )  # [rot:vel] size 6

        return abs(left_contact_force - right_contact_force) + abs(
            right_speed - left_speed
        )

    def orient_reward(self):
        desired_yaw = self.target_velocities[2]
        current_yaw = R.from_matrix(
            np.array(self.data.body("base").xmat).reshape(3, 3)
        ).as_euler("xyz")[2]

        return -((abs(desired_yaw) - abs(current_yaw)) ** 2)

    def follow_xy_target_reward(self):
        x_velocity = self.data.body("base").cvel[3:][0]
        y_velocity = self.data.body("base").cvel[3:][1]
        x_error = abs(self.target_velocities[0] - x_velocity)
        y_error = abs(self.target_velocities[1] - y_velocity)
        return -(x_error + y_error)

    def follow_yaw_target_reward(self):
        yaw_velocity = self.data.body("base").cvel[:3][2]
        yaw_error = abs(self.target_velocities[2] - yaw_velocity)
        return -yaw_error

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

    def feet_spacing_reward(self):
        target_spacing = 0.12
        left_pos = self.data.body("left_foot").xpos
        right_pos = self.data.body("right_foot").xpos
        spacing = np.linalg.norm(left_pos - right_pos)
        return np.exp(-10 * (spacing - target_spacing) ** 2)

    def both_feet_on_the_ground_reward(self):
        elapsed = self.data.time - self.last_time_both_feet_on_the_ground
        return -(elapsed**2)

    def step(self, a):
        t = self.data.time
        dt = t - self.prev_t

        if self.startup_cooldown > 0:
            self.startup_cooldown -= dt
            self.do_simulation(self.init_pos + self.init_pos_noise, FRAME_SKIP)
            reward = 0
            self.last_time_both_feet_on_the_ground = t
        else:
            self.right_foot_contact = check_contact(
                self.data, self.model, "right_foot", "floor"
            )
            self.left_foot_contact = check_contact(
                self.data, self.model, "left_foot", "floor"
            )

            if self.right_foot_contact and self.left_foot_contact:
                self.last_time_both_feet_on_the_ground = t

            # We want to learn deltas from the initial position
            a += self.init_pos

            # Maybe use that too :)
            current_ctrl = self.data.ctrl.copy()
            delta_max = 0.05
            a = np.clip(a, current_ctrl - delta_max, current_ctrl + delta_max)

            self.do_simulation(a, FRAME_SKIP)

            reward = (
                0.1 * self.support_flying_reward()
                + 0.15 * self.follow_xy_target_reward()
                + 0.15 * self.follow_yaw_target_reward()
                + 0.15 * self.height_reward()
                + 0.05 * self.upright_reward()
                + 0.05 * self.action_reward(a)
                + 0.05 * self.torque_reward()
                + 0.05 * self.feet_spacing_reward()
                + 0.05 * self.both_feet_on_the_ground_reward()
            )
            self.cumulated_reward += reward

        ob = self._get_obs()

        if self.render_mode == "human":
            if self.startup_cooldown <= 0:
                print("support flying reward: ", 0.1 * self.support_flying_reward())
                print("Follow xy target reward: ", 0.5 * self.follow_xy_target_reward())
                print(
                    "Follow yaw target reward: ", 0.5 * self.follow_yaw_target_reward()
                )
                print("Height reward: ", 0.15 * self.height_reward())
                print("Upright reward: ", 0.05 * self.upright_reward())
                print("Action reward: ", 0.05 * self.action_reward(a))
                print("Torque reward: ", 0.05 * self.torque_reward())
                print("Feet spacing reward: ", 0.05 * self.feet_spacing_reward())
                print(
                    "Both feet on the ground reward: ",
                    0.05 * self.both_feet_on_the_ground_reward(),
                )
                print("TARGET : ", self.target_velocities)
                print("===")
            self.render()

        self.prev_t = t
        self.prev_action = a.copy()
        self.prev_torque = self.data.qfrc_actuator.copy()

        return (ob, reward, self.is_terminated(), False, {})  # terminated  # truncated

    def reset_model(self):
        self.prev_t = self.data.time
        self.startup_cooldown = 1.0
        print("CUMULATED REWARD: ", self.cumulated_reward)
        self.cumulated_reward = 0.0
        self.last_time_both_feet_on_the_ground = self.data.time

        v_x = np.random.uniform(0.0, 0.05)
        v_y = np.random.uniform(-0.03, 0.03)
        v_theta = np.random.uniform(-0.1, 0.1)
        # self.target_velocities = np.asarray([v_x, v_y, v_theta])  # x, y, yaw
        self.target_velocities = np.asarray([0.05, 0, 0])  # x, y, yaw

        self.prev_action = np.zeros(self.nb_dofs)
        self.prev_torque = np.zeros(self.nb_dofs)

        self.goto_init()

        self.set_state(self.data.qpos, self.data.qvel)
        return self._get_obs()

    def goto_init(self):
        self.data.qvel[:] = np.zeros(len(self.data.qvel[:]))
        self.init_pos_noise = np.random.uniform(-0.01, 0.01, self.nb_dofs)
        self.data.qpos[7 : 7 + self.nb_dofs] = self.init_pos + self.init_pos_noise
        self.data.qpos[2] = 0.15
        self.data.qpos[3 : 3 + 4] = [1, 0, 0.08, 0]

        self.data.ctrl[:] = self.init_pos

    def get_clock_signal(self):
        a = np.sin(2 * np.pi * (self.data.time % self.walk_period) / self.walk_period)
        b = np.cos(2 * np.pi * (self.data.time % self.walk_period) / self.walk_period)
        return [a, b]

    def _get_obs(self):
        joints_rotations = self.data.qpos[7 : 7 + self.nb_dofs]
        joints_velocities = self.data.qvel[6 : 6 + self.nb_dofs]

        # TODO this is imu, add noise to it later
        angular_velocity = self.data.body("base").cvel[:3]
        linear_velocity = self.data.body("base").cvel[3:]

        return np.concatenate(
            [
                joints_rotations,
                joints_velocities,
                angular_velocity,
                linear_velocity,
                self.target_velocities,
                [self.left_foot_contact, self.right_foot_contact],
                self.get_clock_signal(),
            ]
        )

from os import path

import numpy as np
from gymnasium import spaces, utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class BDXEnv(MujocoEnv, utils.EzPickle):
    """
    BDX Env
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.001,
        reset_noise_scale=0.1,
        **kwargs
    ):
        utils.EzPickle.__init__(
            self, forward_reward_weight, ctrl_cost_weight, reset_noise_scale, **kwargs
        )

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self.startup_cooldown = 1.0
        self.prev_t = 0

        # TODO shape
        observation_space = Box(low=-np.inf, high=np.inf, shape=(43,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            "/home/antoine/MISC/mini_BDX/mini_bdx/robots/bdx/scene.xml",
            20,  # frame_skip
            observation_space=observation_space,
            **kwargs
        )

        self.init_qpos = np.array(
            [
                0,
                0,
                0.15,
                1,
                0,
                0.08,
                0,
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

        # TODO
        self.init_foot_quat = np.array(
            [-0.24135508, -0.24244352, -0.66593612, 0.66294642]
        )

        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        self._action_lower_bounds, self._action_upper_bounds = bounds.T
        self.action_space = spaces.Box(
            low=-np.ones_like(self._action_lower_bounds),
            high=np.ones_like(self._action_upper_bounds),
            dtype=np.float32,
        )

    def control_cost(self, true_action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(true_action))
        return control_cost

    @staticmethod
    def quat_distance(q1, q2):
        return 2.0 * np.arccos(max(min(np.sum(q1 * q2), 1 - 1e-10), -1 + 1e-10))

    def step(self, action):

        t = self.data.time
        dt = t - self.prev_t

        x_position_before = (
            2.0 * self.data.qpos[0]
            + self.data.body("left_foot").xpos[0]
            + self.data.body("right_foot").xpos[0]
        )
        true_action = (action + 1.0) / 2.0 * (
            self._action_upper_bounds - self._action_lower_bounds
        ) + self._action_lower_bounds

        if self.startup_cooldown > 0:
            self.startup_cooldown -= dt
        if self.startup_cooldown <= 0:
            self.do_simulation(true_action, self.frame_skip)
        else:
            self.do_simulation(self.init_qpos[7:], self.frame_skip)
            reward = 0

        x_position_after = (
            2.0 * self.data.qpos[0]
            + self.data.body("left_foot").xpos[0]
            + self.data.body("right_foot").xpos[0]
        )
        x_velocity = 2.0 * (x_position_after - x_position_before) / self.dt
        z_position_after = self.data.qpos[2]

        ctrl_cost = self.control_cost(true_action)

        forward_reward = (
            self._forward_reward_weight * x_velocity
            - np.abs(self.data.qpos[1]) * 10.0
            - self.quat_distance(
                self.init_foot_quat, self.data.body("right_foot").xquat
            )
            * 5
            - self.quat_distance(self.init_foot_quat, self.data.body("left_foot").xquat)
            * 5
        )
        observation = self._get_obs()
        terminated = False
        # left_foot_contact_force = np.sum(
        #     np.square(self.data.cfrc_ext[12] + self.data.cfrc_ext[13])
        # )
        # right_foot_contact_force = np.sum(
        #     np.square(self.data.cfrc_ext[24] + self.data.cfrc_ext[25])
        # )
        left_foot_contact_force = np.sum(np.square(self.data.cfrc_ext[11]))
        right_foot_contact_force = np.sum(np.square(self.data.cfrc_ext[22]))

        # print(left_foot_contact_force, right_foot_contact_force)
        reward = forward_reward - ctrl_cost

        if (
            self.data.body("left_foot").xpos[2] > 0.4
            or self.data.body("right_foot").xpos[2] > 0.4
            or np.abs(
                self.data.body("left_foot").xpos[0]
                - self.data.body("right_foot").xpos[0]
            )
            > 1.0
        ):  # constraint on step length:
            reward = reward - 20.0

        if left_foot_contact_force < 500.0 and right_foot_contact_force < 500.0:
            reward = reward - 20.0

        if z_position_after < 0.07:
            reward = reward - 200.0
            terminated = True

        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }

        if self.render_mode == "human":
            self.render()

        self.prev_t = t
        return observation, reward, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        self.prev_t = self.data.time
        self.startup_cooldown = 1.0
        qpos = self.init_qpos
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

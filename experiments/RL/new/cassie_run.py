# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

# ================================================================================

# Part of this file is derived from half_cheetah_v4.py in Open AI gym, with the
# following licence:

# The MIT License

# Copyright (c) 2016 OpenAI (https://openai.com)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# ================================================================================

__credits__ = ["Nicolas Perrin-Gilbert", "Rushiv Arora"]

from os import path

import numpy as np
from gymnasium import spaces, utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class CassieRunEnv(MujocoEnv, utils.EzPickle):
    """
    Cassie Env
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
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

        observation_space = Box(low=-np.inf, high=np.inf, shape=(67,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            "/home/antoine/MISC/gym-cassie-run/gym_cassie_run/env/assets/cassie.xml",
            20,  # frame_skip
            observation_space=observation_space,
            **kwargs
        )

        self.init_qpos = np.array(
            [
                0,
                0,
                1.0059301,
                1,
                0,
                0,
                0,
                0.00449956,
                0,
                0.497301,
                0.97861,
                -0.0164104,
                0.0177766,
                -0.204298,
                -1.1997,
                0,
                1.42671,
                -2.25907e-06,
                -1.52439,
                1.50645,
                -1.59681,
                -0.00449956,
                0,
                0.497301,
                0.97874,
                0.0038687,
                -0.0151572,
                -0.204509,
                -1.1997,
                0,
                1.42671,
                0,
                -1.52439,
                1.50645,
                -1.59681,
            ]
        )

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
        x_position_before = (
            2.0 * self.data.qpos[0]
            + self.data.body("left-foot").xpos[0]
            + self.data.body("right-foot").xpos[0]
        )
        true_action = (action + 1.0) / 2.0 * (
            self._action_upper_bounds - self._action_lower_bounds
        ) + self._action_lower_bounds
        self.do_simulation(true_action, self.frame_skip)

        x_position_after = (
            2.0 * self.data.qpos[0]
            + self.data.body("left-foot").xpos[0]
            + self.data.body("right-foot").xpos[0]
        )
        x_velocity = 2.0 * (x_position_after - x_position_before) / self.dt
        z_position_after = self.data.qpos[2]

        ctrl_cost = self.control_cost(true_action)

        forward_reward = (
            self._forward_reward_weight * x_velocity
            - np.abs(self.data.qpos[1]) * 10.0
            - self.quat_distance(
                self.init_foot_quat, self.data.body("right-foot").xquat
            )
            * 5
            - self.quat_distance(self.init_foot_quat, self.data.body("left-foot").xquat)
            * 5
        )
        observation = self._get_obs()
        terminated = False
        left_foot_contact_force = np.sum(
            np.square(self.data.cfrc_ext[12] + self.data.cfrc_ext[13])
        )
        right_foot_contact_force = np.sum(
            np.square(self.data.cfrc_ext[24] + self.data.cfrc_ext[25])
        )

        reward = forward_reward - ctrl_cost

        if (
            self.data.body("left-foot").xpos[2] > 0.4
            or self.data.body("right-foot").xpos[2] > 0.4
            or np.abs(
                self.data.body("left-foot").xpos[0]
                - self.data.body("right-foot").xpos[0]
            )
            > 1.0
        ):  # constraint on step length:
            reward = reward - 20.0

        if left_foot_contact_force < 500.0 and right_foot_contact_force < 500.0:
            reward = reward - 20.0

        if z_position_after < 0.8:
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
        return observation, reward, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
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

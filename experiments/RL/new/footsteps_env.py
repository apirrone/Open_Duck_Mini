# Inspired by https://arxiv.org/pdf/2207.12644
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from mini_bdx.placo_walk_engine import PlacoWalkEngine

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
                    *(
                        -10 * np.ones(8)
                    ),  # next two footsteps [x, y, z, theta, x, y, z, theta]
                    *(-np.pi * np.ones(2)),  # clock signal
                ]
            ),
            np.array(
                [
                    *(np.pi * np.ones(self.nb_dofs)),  # joints_rotations
                    *(10 * np.ones(self.nb_dofs)),  # joints_velocities
                    *(10 * np.ones(3)),  # angular_velocity
                    *(10 * np.ones(3)),  # linear_velocity
                    *(
                        10 * np.ones(8)
                    ),  # next two footsteps [x, y, z, theta, x, y, z, theta]
                    *(np.pi * np.ones(2)),  # clock signal
                ]
            ),
        )

        self.startup_cooldown = 1.0

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
            self.data.body("base").xpos[2] < 0.08 or np.dot(upright, Z_vec) <= 0.2
        )  # base z is below 0.08m or base has more than 90 degrees of tilt

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

            reward = (
                0.05  # time reward
                + 0.1 * self.walking_height_reward()
                + 0.1 * self.upright_reward()
                + 0.1 * self.velocity_tracking_reward()
                + 0.01 * self.smoothness_reward2()
            )

        ob = self._get_obs()

        if self.render_mode == "human":
            self.render()

        self.prev_t = t

        # self.viz.display(self.pwe.robot.state.q)
        return (ob, reward, self.is_terminated(), False, {})  # terminated  # truncated

    def reset_model(self):
        self.prev_t = self.data.time
        self.startup_cooldown = 1.0
        self.pwe.reset()

        self.goto_init()

        self.set_state(self.data.qpos, self.data.qvel)
        return self._get_obs()

    def goto_init(self):
        self.data.qvel[:] = np.zeros(len(self.data.qvel[:]))
        self.data.qpos[7 : 7 + self.nb_dofs] = self.init_pos
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

        next_two_footsteps = self.pwe.get_footsteps_in_robot_frame()[:2]

        return np.concatenate(
            [
                joints_rotations,
                joints_velocities,
                angular_velocity,
                linear_velocity,
                next_two_footsteps,
                self.get_clock_signal(),
            ]
        )

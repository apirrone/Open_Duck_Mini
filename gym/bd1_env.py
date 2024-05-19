import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


class BD1Env(MujocoEnv, utils.EzPickle):
    """
    ## Action space

    | Num  | Action                                                            | Control Min | Control Max | Name (in corresponding XML file) | Joint    | Unit         |
    | ---- | ----------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | -------- |------------- |
    | 0    | Apply torque on right_hip_yaw                                     | -0.58       | 0.58        | right_hip_yaw                    | cylinder | torque (N m) |
    | 1    | Apply torque on right_hip_roll                                    | -0.58       | 0.58        | right_hip_roll                   | cylinder | torque (N m) |
    | 2    | Apply torque on right_hip_pitch                                   | -0.58       | 0.58        | right_hip_pitch                  | cylinder | torque (N m) |
    | 3    | Apply torque on right_knee_pitch                                  | -0.58       | 0.58        | right_knee_pitch                 | cylinder | torque (N m) |
    | 4    | Apply torque on right_ankle_pitch                                 | -0.58       | 0.58        | right_ankle_pitch                | cylinder | torque (N m) |
    | 5    | Apply torque on left_hip_yaw                                      | -0.58       | 0.58        | left_hip_yaw                     | cylinder | torque (N m) |
    | 6    | Apply torque on left_hip_roll                                     | -0.58       | 0.58        | left_hip_roll                    | cylinder | torque (N m) |
    | 7    | Apply torque on left_hip_pitch                                    | -0.58       | 0.58        | left_hip_pitch                   | cylinder | torque (N m) |
    | 8    | Apply torque on left_knee_pitch                                   | -0.58       | 0.58        | left_knee_pitch                  | cylinder | torque (N m) |
    | 9    | Apply torque on left_ankle_pitch                                  | -0.58       | 0.58        | left_ankle_pitch                 | cylinder | torque (N m) |

    ## Observation space

    | Num | Observation                                              | Min  | Max | Name (in corresponding XML file) | Joint    | Unit                     |
    | --- | -------------------------------------------------------- | ---- | --- | -------------------------------- | -------- | ------------------------ |
    | 0   | Rotation right_hip_yaw                                   | -Inf | Inf | right_hip_yaw                    | cylinder | angle (rad)              |
    | 1   | Rotation right_hip_roll                                  | -Inf | Inf | right_hip_roll                   | cylinder | angle (rad)              |
    | 2   | Rotation right_hip_pitch                                 | -Inf | Inf | right_hip_pitch                  | cylinder | angle (rad)              |
    | 3   | Rotation right_knee_pitch                                | -Inf | Inf | right_knee_pitch                 | cylinder | angle (rad)              |
    | 4   | Rotation right_ankle_pitch                               | -Inf | Inf | right_ankle_pitch                | cylinder | angle (rad)              |
    | 5   | Rotation left_hip_yaw                                    | -Inf | Inf | left_hip_yaw                     | cylinder | angle (rad)              |
    | 6   | Rotation left_hip_roll                                   | -Inf | Inf | left_hip_roll                    | cylinder | angle (rad)              |
    | 7   | Rotation left_hip_pitch                                  | -Inf | Inf | left_hip_pitch                   | cylinder | angle (rad)              |
    | 8   | Rotation left_knee_pitch                                 | -Inf | Inf | left_knee_pitch                  | cylinder | angle (rad)              |
    | 9   | Rotation left_ankle_pitch                                | -Inf | Inf | left_ankle_pitch                 | cylinder | angle (rad)              |
    | 10  | velocity of right_hip_yaw                                | -Inf | Inf | right_hip_yaw                    | cylinder | speed (rad/s)            |
    | 11  | velocity of right_hip_roll                               | -Inf | Inf | right_hip_roll                   | cylinder | speed (rad/s)            |
    | 12  | velocity of right_hip_pitch                              | -Inf | Inf | right_hip_pitch                  | cylinder | speed (rad/s)            |
    | 13  | velocity of right_knee_pitch                             | -Inf | Inf | right_knee_pitch                 | cylinder | speed (rad/s)            |
    | 14  | velocity of right_ankle_pitch                            | -Inf | Inf | right_ankle_pitch                | cylinder | speed (rad/s)            |
    | 15  | velocity of left_hip_yaw                                 | -Inf | Inf | left_hip_yaw                     | cylinder | speed (rad/s)            |
    | 16  | velocity of left_hip_roll                                | -Inf | Inf | left_hip_roll                    | cylinder | speed (rad/s)            |
    | 17  | velocity of left_hip_pitch                               | -Inf | Inf | left_hip_pitch                   | cylinder | speed (rad/s)            |
    | 18  | velocity of left_knee_pitch                              | -Inf | Inf | left_knee_pitch                  | cylinder | speed (rad/s)            |
    | 19  | velocity of left_ankle_pitch                             | -Inf | Inf | left_ankle_pitch                 | cylinder | speed (rad/s)            |
    | 20  | x component of up vector                                 | -Inf | Inf |                                  |          | vec                      |
    | 21  | y component of up vector                                 | -Inf | Inf |                                  |          | vec                      |
    | 22  | z component of up vector                                 | -Inf | Inf |                                  |          | vec                      |

    # TODO give angle around z axis to target
    # TODO add velocities
    # below for later
    | 10  | x component of 2D vector to the target                   | -Inf | Inf |                                  |          | position (m)             |
    | 11  | y component of 2D vector to the target                   | -Inf | Inf |                                  |          | position (m)             |
    | x13 | x position of the center of mass                         | -Inf | Inf |                                  |          | position (m)             |
    | x14 | y position of the center of mass                         | -Inf | Inf |                                  |          | position (m)             |
    | x15 | z position of the center of mass                         | -Inf | Inf |                                  |          | position (m)             |
    | x16 | roll angle of the base                                   | -Inf | Inf |                                  |          | angle (rad)              |
    | x17 | pitch angle of the base                                  | -Inf | Inf |                                  |          | angle (rad)              |
    | x18 | yaw angle of the base                                    | -Inf | Inf |                                  |          | angle (rad)              |

    data.qpos[-7:-4]
    data.qvel[8 : 8 + 10]
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
        healthy_z_range=[0.12, 0.35],
        healthy_reward=5.0,
        reached_goal_reward=20.0,
        **kwargs
    ):
        utils.EzPickle.__init__(
            self, healthy_z_range, healthy_reward, reached_goal_reward, **kwargs
        )
        observation_space = Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float64)
        self.goal_pos = np.asarray([0, 0, 0])
        self._healthy_z_range = healthy_z_range
        self._healthy_reward = healthy_reward
        self._reached_goal_reward = reached_goal_reward
        MujocoEnv.__init__(
            self,
            "/home/antoine/MISC/mini_BD1/robots/bd1/scene.xml",
            5,
            observation_space=observation_space,
            **kwargs,
        )

    def is_healthy(self) -> bool:
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z
        return bool(is_healthy)

    def is_terminated(self) -> bool:
        return bool(not self.is_healthy() or self.has_reached_goal())

    def get_healthy_reward(self) -> bool:
        return self.is_healthy() * self._healthy_reward

    def has_reached_goal(self) -> bool:
        return bool(
            (
                np.linalg.norm(
                    self.get_body_com("base")[:2] - self.get_body_com("goal")[:2]
                )
                < 0.15
            )
        )

    def get_reached_goal_reward(self):
        return self.has_reached_goal() * self._reached_goal_reward

    def step(self, a):

        reward_dist = -np.linalg.norm(
            self.get_body_com("base")[:2] - self.get_body_com("goal")[:2]
        )
        reward_ctrl = -np.square(a).sum()

        # reward = (
        #     reward_dist
        #     + 0.1 * reward_ctrl
        #     + self.get_healthy_reward()
        #     + self.get_reached_goal_reward()
        # )

        reward = self.get_healthy_reward()  # trying to make the robot stand

        self.do_simulation(a, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()
        # terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
        #         which can be positive or negative. An example is reaching the goal state or moving into the lava from
        #         the Sutton and Barton, Gridworld. If true, the user needs to call :meth:`reset`.
        # truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
        #         Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
        #         Can be used to end the episode prematurely before a terminal state is reached.
        #         If true, the user needs to call :meth:`reset`.

        if self.is_terminated():
            print(
                "terminated because",
                "not healthy" if not self.is_healthy() else "reached goal",
            )
            # self.reset()  # not needed because autoreset is True in register

        return (
            ob,
            reward,
            self.is_terminated(),  # terminated
            False,  # truncated
            dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl),
        )

    def reset_model(self):
        # TODO maybe try to set the robot to a good starting position, knees bent etc
        qpos = self.init_qpos

        # goal pos is a random point a circle of radius 1 around the origin
        angle = np.random.uniform(0, 2 * np.pi)
        radius = 1  # m
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

        self.goal_pos = np.asarray(
            [
                x,
                y,
                0.01,
            ]
        )
        qpos[-7:-4] = self.goal_pos
        # TODO try adding noise to qpos and qvel
        # qpos = self.init_qpos + self.np_random.uniform(
        #     low=noise_low, high=noise_high, size=self.model.nq
        # )
        # qvel = self.init_qvel + self.np_random.uniform(
        #     low=noise_low, high=noise_high, size=self.model.nv
        # )

        self.set_state(qpos, self.init_qvel)
        return self._get_obs()

    def _get_obs(self):
        # vec = self.get_body_com("goal")[:2] - self.get_body_com("base")[:2]

        rotations = self.data.qpos[8 : 8 + 10]
        velocities = self.data.qvel[8 : 8 + 10]
        # print(rotations)
        # print(velocities)

        rot = np.array(self.data.body("base").xmat).reshape(3, 3)
        Z_vec = rot[:, 2]
        return np.concatenate([rotations, velocities, Z_vec])

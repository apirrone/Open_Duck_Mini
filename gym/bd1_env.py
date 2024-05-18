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
    | 10  | x component of 2D vector to the target                   | -Inf | Inf |                                  |          | position (m)             |
    | 11  | y component of 2D vector to the target                   | -Inf | Inf |                                  |          | position (m)             |

    # TODO give angle around z axis to target
    # below for later
    | x13 | x position of the center of mass                         | -Inf | Inf |                                  |          | position (m)             |
    | x14 | y position of the center of mass                         | -Inf | Inf |                                  |          | position (m)             |
    | x15 | z position of the center of mass                         | -Inf | Inf |                                  |          | position (m)             |
    | x16 | roll angle of the base                                   | -Inf | Inf |                                  |          | angle (rad)              |
    | x17 | pitch angle of the base                                  | -Inf | Inf |                                  |          | angle (rad)              |
    | x18 | yaw angle of the base                                    | -Inf | Inf |                                  |          | angle (rad)              |

    data.qpos[-7:-4]

    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64)
        self.goal_pos = np.asarray([0, 0, 0])
        MujocoEnv.__init__(
            self,
            "/home/antoine/MISC/mini_BD1/robots/bd1/scene.xml",
            5,
            observation_space=observation_space,
            **kwargs,
        )

    def step(self, a):

        reward_dist = -np.linalg.norm(
            self.get_body_com("base")[:2] - self.get_body_com("goal")[:2]
        )
        reward_ctrl = -np.square(a).sum()

        reward = reward_dist + 0.1 * reward_ctrl

        # Add height of com to reward to encourage walking

        self.do_simulation(a, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()
        return (
            ob,
            reward,
            False,
            False,
            dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl),
        )

    def reset_model(self):
        # TODO maybe try to set the robot to a good starting position, knees bent etc
        qpos = self.init_qpos

        self.goal_pos = np.asarray(
            [
                self.np_random.uniform(low=-0.6, high=0.6, size=1)[0],
                self.np_random.uniform(low=-0.6, high=0.6, size=1)[0],
                0.01,
            ]
        )
        qpos[-7:-4] = self.goal_pos

        self.set_state(qpos, self.init_qvel)
        return self._get_obs()

    def _get_obs(self):
        vec = self.get_body_com("goal")[:2] - self.get_body_com("base")[:2]
        return np.concatenate(
            [
                self.data.qpos.flat[:10],
                vec,
            ]
        )

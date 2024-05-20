import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

init_pos = {
    "right_hip_yaw": 0,
    "right_hip_roll": 0,
    "right_hip_pitch": np.deg2rad(45),
    "right_knee_pitch": np.deg2rad(-90),
    "right_ankle_pitch": np.deg2rad(45),
    "left_hip_yaw": 0,
    "left_hip_roll": 0,
    "left_hip_pitch": np.deg2rad(45),
    "left_knee_pitch": np.deg2rad(-90),
    "left_ankle_pitch": np.deg2rad(45),
}


class BD1Env(MujocoEnv, utils.EzPickle):
    """
    ## Action space

    | Num  | Action                                                            | Control Min | Control Max | Name (in corresponding XML file) | Joint    | Unit         |
    | ---- | ----------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | -------- |------------- |
    | 0    | Set position of right_hip_yaw                                     | -0.58TODO   | 0.58TODO    | right_hip_yaw                    | cylinder | pos (rad)    |
    | 1    | Set position of right_hip_roll                                    | -0.58TODO   | 0.58TODO    | right_hip_roll                   | cylinder | pos (rad)    |
    | 2    | Set position of right_hip_pitch                                   | -0.58TODO   | 0.58TODO    | right_hip_pitch                  | cylinder | pos (rad)    |
    | 3    | Set position of right_knee_pitch                                  | -0.58TODO   | 0.58TODO    | right_knee_pitch                 | cylinder | pos (rad)    |
    | 4    | Set position of right_ankle_pitch                                 | -0.58TODO   | 0.58TODO    | right_ankle_pitch                | cylinder | pos (rad)    |
    | 5    | Set position of left_hip_yaw                                      | -0.58TODO   | 0.58TODO    | left_hip_yaw                     | cylinder | pos (rad)    |
    | 6    | Set position of left_hip_roll                                     | -0.58TODO   | 0.58TODO    | left_hip_roll                    | cylinder | pos (rad)    |
    | 7    | Set position of left_hip_pitch                                    | -0.58TODO   | 0.58TODO    | left_hip_pitch                   | cylinder | pos (rad)    |
    | 8    | Set position of left_knee_pitch                                   | -0.58TODO   | 0.58TODO    | left_knee_pitch                  | cylinder | pos (rad)    |
    | 9    | Set position of left_ankle_pitch                                  | -0.58TODO   | 0.58TODO    | left_ankle_pitch                 | cylinder | pos (rad)    |

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
    | 20  | x component of up vector                                 | -Inf | Inf |                                  |          |                          |
    | 21  | y component of up vector                                 | -Inf | Inf |                                  |          |                          |
    | 22  | z component of up vector                                 | -Inf | Inf |                                  |          |                          |
    | 23  | current x linear velocity                                | -Inf | Inf |                                  |          |                          |
    | 24  | current y linear velocity                                | -Inf | Inf |                                  |          |                          |
    | 25  | current yaw angular velocity                             | -Inf | Inf |                                  |          |                          |
    | 26  | current x target linear velocity                         | -Inf | Inf |                                  |          |                          |
    | 27  | current y target linear velocity                         | -Inf | Inf |                                  |          |                          |
    | 28  | current yaw target angular velocity                      | -Inf | Inf |                                  |          |                          |

    | 29  | t-1 right_hip_yaw rotation error                         | -Inf | Inf |                                  |          |                          |
    | 30  | t-1 right_hip_roll rotation error                        | -Inf | Inf |                                  |          |                          |
    | 31  | t-1 right_hip_pitch rotation error                       | -Inf | Inf |                                  |          |                          |
    | 32  | t-1 right_knee_pitch rotation error                      | -Inf | Inf |                                  |          |                          |
    | 33  | t-1 right_ankle_pitch rotation error                     | -Inf | Inf |                                  |          |                          |
    | 34  | t-1 left_hip_yaw rotation error                          | -Inf | Inf |                                  |          |                          |
    | 35  | t-1 left_hip_roll rotation error                         | -Inf | Inf |                                  |          |                          |
    | 36  | t-1 left_hip_pitch rotation error                        | -Inf | Inf |                                  |          |                          |
    | 37  | t-1 left_knee_pitch rotation error                       | -Inf | Inf |                                  |          |                          |
    | 38  | t-1 left_ankle_pitch rotation error                      | -Inf | Inf |                                  |          |                          |
    | 39  | t-2 right_hip_yaw rotation error                         | -Inf | Inf |                                  |          |                          |
    | 40  | t-2 right_hip_roll rotation error                        | -Inf | Inf |                                  |          |                          |
    | 41  | t-2 right_hip_pitch rotation error                       | -Inf | Inf |                                  |          |                          |
    | 42  | t-2 right_knee_pitch rotation error                      | -Inf | Inf |                                  |          |                          |
    | 43  | t-2 right_ankle_pitch rotation error                     | -Inf | Inf |                                  |          |                          |
    | 44  | t-2 left_hip_yaw rotation error                          | -Inf | Inf |                                  |          |                          |
    | 45  | t-2 left_hip_roll rotation error                         | -Inf | Inf |                                  |          |                          |
    | 46  | t-2 left_hip_pitch rotation error                        | -Inf | Inf |                                  |          |                          |
    | 47  | t-2 left_knee_pitch rotation error                       | -Inf | Inf |                                  |          |                          |
    | 48  | t-2 left_ankle_pitch rotation error                      | -Inf | Inf |                                  |          |                          |
    | 49  | t-3 right_hip_yaw rotation error                         | -Inf | Inf |                                  |          |                          |
    | 50  | t-3 right_hip_roll rotation error                        | -Inf | Inf |                                  |          |                          |
    | 51  | t-3 right_hip_pitch rotation error                       | -Inf | Inf |                                  |          |                          |
    | 52  | t-3 right_knee_pitch rotation error                      | -Inf | Inf |                                  |          |                          |
    | 53  | t-3 right_ankle_pitch rotation error                     | -Inf | Inf |                                  |          |                          |
    | 54  | t-3 left_hip_yaw rotation error                          | -Inf | Inf |                                  |          |                          |
    | 55  | t-3 left_hip_roll rotation error                         | -Inf | Inf |                                  |          |                          |
    | 56  | t-3 left_hip_pitch rotation error                        | -Inf | Inf |                                  |          |                          |
    | 57  | t-3 left_knee_pitch rotation error                       | -Inf | Inf |                                  |          |                          |
    | 58  | t-3 left_ankle_pitch rotation error                      | -Inf | Inf |                                  |          |                          |

    # TODO add 1hz sinus ? to help learn the gait


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
        observation_space = Box(low=-np.inf, high=np.inf, shape=(59,), dtype=np.float64)
        self.target_velocity = np.asarray([1, 0, 0])  # x, y, yaw
        self.joint_history_length = 3
        self.joint_error_history = self.joint_history_length * [10 * [0]]
        self.joint_ctrl_history = self.joint_history_length * [10 * [0]]
        MujocoEnv.__init__(
            self,
            "/home/antoine/MISC/mini_BD1/robots/bd1/scene.xml",
            5,
            observation_space=observation_space,
            **kwargs,
        )

    def compute_smoothness_reward(self):
        # Warning, this function only works if the history is 3 :)
        smooth = 0
        t0 = self.joint_ctrl_history[0]
        t_minus1 = self.joint_ctrl_history[1]
        t_minus2 = self.joint_ctrl_history[2]

        for i in range(10):
            smooth += 2.5 * np.square(t0[i] - t_minus1[i]) + 1.5 * np.square(
                t0[i] - 2 * t_minus1[i] + t_minus2[i]
            )

        return -smooth

    def is_terminated(self) -> bool:
        rot = np.array(self.data.body("base").xmat).reshape(3, 3)
        Z_vec = rot[:, 2]
        upright = np.array([0, 0, 1])
        return (
            self.data.body("base").xpos[2] < 0.05 or np.dot(upright, Z_vec) <= 0
        )  # base has more than 90 degrees of tilt

    def step(self, a):
        # https://www.nature.com/articles/s41598-023-38259-7.pdf

        # angular distance to upright position in reward
        Z_vec = np.array(self.data.body("base").xmat).reshape(3, 3)[:, 2]
        upright_reward = np.square(np.dot(np.array([0, 0, 1]), Z_vec))

        walking_height_reward = (
            -np.square((self.get_body_com("base")[2] - 0.12)) * 100
        )  # "normal" walking height is about 0.12m

        current_ctrl = self.data.ctrl
        init_ctrl = np.array(list(init_pos.values()))
        joint_angle_deviation_reward = -np.square(current_ctrl - init_ctrl).sum()

        base_velocity = list(self.data.body("base").cvel[3:][:2]) + [
            self.data.body("base").cvel[:3][2]
        ]
        base_velocity = np.asarray(base_velocity)
        velocity_tracking_reward = np.exp(
            -np.square(base_velocity - self.target_velocity).sum()
        )

        smoothness_reward = self.compute_smoothness_reward()

        reward = (
            0.05  # time reward
            # + 0.2 * walking_height_reward
            + 1 * upright_reward
            + 7 * velocity_tracking_reward
            # + 0.5 * joint_angle_deviation_reward
            + 0.1 * smoothness_reward
        )

        # print("walking_height_reward", walking_height_reward)
        # print("upright_reward", upright_reward)
        # print("velocity_tracking_reward", velocity_tracking_reward)
        # print("joint_angle_deviation_reward", joint_angle_deviation_reward)
        # print("time_reward", 0.05)
        # print("smoothness_reward", smoothness_reward)
        # print("reward", reward)
        # print("---")

        self.do_simulation(a, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()

        if self.is_terminated():
            print(
                "Terminated because too much tilt or com too low.",
            )
            reward -= 100
            # self.reset()  # not needed because autoreset is True in register

        return (
            ob,
            reward,
            self.is_terminated(),  # terminated
            False,  # truncated
            dict(
                walking_height_reward=walking_height_reward,
                upright_reward=upright_reward,
                velocity_tracking_reward=velocity_tracking_reward,
                joint_angle_deviation_reward=joint_angle_deviation_reward,
                time_reward=0.05,
            ),
        )

    def reset_model(self):
        self.goto_init()
        qpos = self.data.qpos
        self.init_qpos = qpos.copy().flatten()

        # Randomize later
        self.target_velocity = np.asarray([1, 0, 0])  # x, y, yaw

        self.set_state(qpos, self.init_qvel)
        return self._get_obs()

    def goto_init(self):
        self.data.qpos[:] = self.init_qpos[:]
        for i, value in enumerate(init_pos.values()):
            self.data.qpos[i + 7] = value

    def _get_obs(self):

        joints_rotations = self.data.qpos[7 : 7 + 10]
        joints_velocities = self.data.qvel[6 : 6 + 10]

        # TODO This is the IMU, add noise to it when trying to go real
        Z_vec = np.array(self.data.body("base").xmat).reshape(3, 3)[:, 2]

        base_velocity = list(self.data.body("base").cvel[3:][:2]) + [
            self.data.body("base").cvel[:3][2]
        ]
        base_velocity = np.asarray(base_velocity)

        joints_error = self.data.ctrl - self.data.qpos[7 : 7 + 10]
        self.joint_error_history.append(joints_error)
        self.joint_error_history = self.joint_error_history[
            -self.joint_history_length :
        ]

        self.joint_ctrl_history.append(self.data.ctrl.copy())
        self.joint_ctrl_history = self.joint_ctrl_history[-self.joint_history_length :]

        return np.concatenate(
            [
                joints_rotations,
                joints_velocities,
                Z_vec,
                base_velocity,
                self.target_velocity,
                np.array(self.joint_error_history).flatten(),
            ]
        )

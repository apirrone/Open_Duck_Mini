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
        observation_space = Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float64)
        self.target_velocity = np.asarray([1, 0, 0])  # x, y, yaw
        MujocoEnv.__init__(
            self,
            "/home/antoine/MISC/mini_BD1/robots/bd1/scene.xml",
            5,
            observation_space=observation_space,
            **kwargs,
        )

    def is_terminated(self) -> bool:
        rot = np.array(self.data.body("base").xmat).reshape(3, 3)
        Z_vec = rot[:, 2]
        upright = np.array([0, 0, 1])
        return np.dot(upright, Z_vec) <= 0  # base has more than 90 degrees of tilt

    def step(self, a):
        # https://www.nature.com/articles/s41598-023-38259-7.pdf
        # ctrl_reward = -0.1 * np.square(a).sum()

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

        # TODO try to add reward for being close to manually set init position ?
        reward = (
            0.05  # time reward
            + 1 * walking_height_reward
            + 1 * upright_reward
            + 5 * velocity_tracking_reward
            + 1 * joint_angle_deviation_reward
        )

        # print("walking_height_reward", walking_height_reward)
        # print("upright_reward", upright_reward)
        # print("velocity_tracking_reward", velocity_tracking_reward)
        # print("joint_angle_deviation_reward", joint_angle_deviation_reward)
        # print("time_reward", 0.05)
        # print("reward", reward)
        # print("---")

        self.do_simulation(a, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()

        if self.is_terminated():
            print(
                "Terminated because too much tilt",
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

        return np.concatenate(
            [
                joints_rotations,
                joints_velocities,
                Z_vec,
                base_velocity,
                self.target_velocity,
            ]
        )

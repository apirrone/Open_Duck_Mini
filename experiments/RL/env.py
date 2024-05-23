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
    "head_pitch1": np.deg2rad(-45),
    "head_pitch2": np.deg2rad(-45),
    "head_yaw": 0,
}


dofs = {
    "right_hip_yaw": 0,
    "right_hip_roll": 1,
    "right_hip_pitch": 2,
    "right_knee_pitch": 3,
    "right_ankle_pitch": 4,
    "left_hip_yaw": 5,
    "left_hip_roll": 6,
    "left_hip_pitch": 7,
    "left_knee_pitch": 8,
    "left_ankle_pitch": 9,
    "head_pitch1": 10,
    "head_pitch2": 11,
    "head_yaw": 12,
}


class BDXEnv(MujocoEnv, utils.EzPickle):
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
    | 9    | Set position of head_pitch1                                       | -0.58TODO   | 0.58TODO    | left_ankle_pitch                 | cylinder | pos (rad)    |
    | 9    | Set position of head_pitch2                                       | -0.58TODO   | 0.58TODO    | left_ankle_pitch                 | cylinder | pos (rad)    |
    | 9    | Set position of head_yaw                                          | -0.58TODO   | 0.58TODO    | left_ankle_pitch                 | cylinder | pos (rad)    |


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
    | 10  | Rotation head_pitch1                                     | -Inf | Inf | head_pitch1                      | cylinder | angle (rad)              |
    | 11  | Rotation head_pitch2                                     | -Inf | Inf | head_pitch2                      | cylinder | angle (rad)              |
    | 12  | Rotation head_yaw                                        | -Inf | Inf | head_yaw                         | cylinder | angle (rad)              |
    | 13  | Velocity of right_hip_yaw                                | -Inf | Inf | right_hip_yaw                    | cylinder | speed (rad/s)            |
    | 14  | Velocity of right_hip_roll                               | -Inf | Inf | right_hip_roll                   | cylinder | speed (rad/s)            |
    | 15  | Velocity of right_hip_pitch                              | -Inf | Inf | right_hip_pitch                  | cylinder | speed (rad/s)            |
    | 16  | Velocity of right_knee_pitch                             | -Inf | Inf | right_knee_pitch                 | cylinder | speed (rad/s)            |
    | 17  | Velocity of right_ankle_pitch                            | -Inf | Inf | right_ankle_pitch                | cylinder | speed (rad/s)            |
    | 18  | Velocity of left_hip_yaw                                 | -Inf | Inf | left_hip_yaw                     | cylinder | speed (rad/s)            |
    | 19  | Velocity of left_hip_roll                                | -Inf | Inf | left_hip_roll                    | cylinder | speed (rad/s)            |
    | 20  | Velocity of left_hip_pitch                               | -Inf | Inf | left_hip_pitch                   | cylinder | speed (rad/s)            |
    | 21  | Velocity of left_knee_pitch                              | -Inf | Inf | left_knee_pitch                  | cylinder | speed (rad/s)            |
    | 22  | Velocity of left_ankle_pitch                             | -Inf | Inf | left_ankle_pitch                 | cylinder | speed (rad/s)            |
    | 23  | Velocity of head_pitch1                                  | -Inf | Inf | head_pitch1                      | cylinder | speed (rad/s)            |
    | 24  | Velocity of head_pitch2                                  | -Inf | Inf | head_pitch2                      | cylinder | speed (rad/s)            |
    | 25  | Velocity of head_yaw                                     | -Inf | Inf | head_yaw                         | cylinder | speed (rad/s)            |


    | 26  | roll angular velocity                                    | -Inf | Inf |                                  |          |                          |e
    | 27  | pitch angular velocity                                   | -Inf | Inf |                                  |          |                          |
    | 28  | yaw angular velocity                                     | -Inf | Inf |                                  |          |                          |
    | 29  | current x linear velocity                                | -Inf | Inf |                                  |          |                          |
    | 30  | current y linear velocity                                | -Inf | Inf |                                  |          |                          |
    | 31  | current z linear velocity                                | -Inf | Inf |                                  |          |                          |
    | 32  | current x target linear velocity                         | -Inf | Inf |                                  |          |                          |
    | 33  | current y target linear velocity                         | -Inf | Inf |                                  |          |                          |
    | 34  | current yaw target angular velocity                      | -Inf | Inf |                                  |          |                          |

    | 35  | t-1 right_hip_yaw rotation error                         | -Inf | Inf |                                  |          |                          |
    | 36  | t-1 right_hip_roll rotation error                        | -Inf | Inf |                                  |          |                          |
    | 37  | t-1 right_hip_pitch rotation error                       | -Inf | Inf |                                  |          |                          |
    | 38  | t-1 right_knee_pitch rotation error                      | -Inf | Inf |                                  |          |                          |
    | 39  | t-1 right_ankle_pitch rotation error                     | -Inf | Inf |                                  |          |                          |
    | 40  | t-1 left_hip_yaw rotation error                          | -Inf | Inf |                                  |          |                          |
    | 41  | t-1 left_hip_roll rotation error                         | -Inf | Inf |                                  |          |                          |
    | 42  | t-1 left_hip_pitch rotation error                        | -Inf | Inf |                                  |          |                          |
    | 43  | t-1 left_knee_pitch rotation error                       | -Inf | Inf |                                  |          |                          |
    | 44  | t-1 left_ankle_pitch rotation error                      | -Inf | Inf |                                  |          |                          |
    | 45  | t-1 head_pitch1 rotation error                           | -Inf | Inf |                                  |          |                          |
    | 46  | t-1 head_pitch2 rotation error                           | -Inf | Inf |                                  |          |                          |
    | 47  | t-1 head_yaw rotation error                              | -Inf | Inf |                                  |          |                          |
    | 48  | t-2 right_hip_yaw rotation error                         | -Inf | Inf |                                  |          |                          |
    | 49  | t-2 right_hip_roll rotation error                        | -Inf | Inf |                                  |          |                          |
    | 50  | t-2 right_hip_pitch rotation error                       | -Inf | Inf |                                  |          |                          |
    | 51  | t-2 right_knee_pitch rotation error                      | -Inf | Inf |                                  |          |                          |
    | 52  | t-2 right_ankle_pitch rotation error                     | -Inf | Inf |                                  |          |                          |
    | 53  | t-2 left_hip_yaw rotation error                          | -Inf | Inf |                                  |          |                          |
    | 54  | t-2 left_hip_roll rotation error                         | -Inf | Inf |                                  |          |                          |
    | 55  | t-2 left_hip_pitch rotation error                        | -Inf | Inf |                                  |          |                          |
    | 56  | t-2 left_knee_pitch rotation error                       | -Inf | Inf |                                  |          |                          |
    | 57  | t-2 left_ankle_pitch rotation error                      | -Inf | Inf |                                  |          |                          |
    | 58  | t-2 head_pitch1 rotation error                           | -Inf | Inf |                                  |          |                          |
    | 59  | t-2 head_pitch2 rotation error                           | -Inf | Inf |                                  |          |                          |
    | 60  | t-2 head_yaw rotation error                              | -Inf | Inf |                                  |          |                          |
    | 61  | t-3 right_hip_yaw rotation error                         | -Inf | Inf |                                  |          |                          |
    | 62  | t-3 right_hip_roll rotation error                        | -Inf | Inf |                                  |          |                          |
    | 63  | t-3 right_hip_pitch rotation error                       | -Inf | Inf |                                  |          |                          |
    | 64  | t-3 right_knee_pitch rotation error                      | -Inf | Inf |                                  |          |                          |
    | 65  | t-3 right_ankle_pitch rotation error                     | -Inf | Inf |                                  |          |                          |
    | 66  | t-3 left_hip_yaw rotation error                          | -Inf | Inf |                                  |          |                          |
    | 67  | t-3 left_hip_roll rotation error                         | -Inf | Inf |                                  |          |                          |
    | 68  | t-3 left_hip_pitch rotation error                        | -Inf | Inf |                                  |          |                          |
    | 69  | t-3 left_knee_pitch rotation error                       | -Inf | Inf |                                  |          |                          |
    | 70  | t-3 left_ankle_pitch rotation error                      | -Inf | Inf |                                  |          |                          |
    | 71  | t-3 head_pitch1 rotation error                           | -Inf | Inf |                                  |          |                          |
    | 72  | t-3 head_pitch2 rotation error                           | -Inf | Inf |                                  |          |                          |
    | 73  | t-3 head_yaw rotation error                              | -Inf | Inf |                                  |          |                          |

    | 74  | sinus                                                    | -Inf | Inf |                                  |          |                          |


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
        observation_space = Box(low=-np.inf, high=np.inf, shape=(75,), dtype=np.float64)
        self.target_velocity = np.asarray([1, 0, 0])  # x, y, yaw
        self.joint_history_length = 3
        self.joint_error_history = self.joint_history_length * [13 * [0]]
        self.joint_ctrl_history = self.joint_history_length * [13 * [0]]
        MujocoEnv.__init__(
            self,
            "/home/antoine/MISC/mini_BDX/mini_bdx/robots/bdx/scene.xml",
            5,
            observation_space=observation_space,
            **kwargs,
        )

    def sample_walk(self):
        freq = 2

        s1 = np.sin(2 * np.pi * freq * self.data.time)
        s2 = np.sin(2 * np.pi * freq * self.data.time + np.pi)

        targets = init_pos.copy()
        targets["right_hip_pitch"] = init_pos["right_hip_pitch"] + s1
        targets["right_knee_pitch"] = init_pos["right_knee_pitch"] - s1

        targets["left_hip_pitch"] = init_pos["left_hip_pitch"] + s2
        targets["left_knee_pitch"] = init_pos["left_knee_pitch"] - s2

        return targets

    def compute_smoothness_reward(self):
        # Warning, this function only works if the history is 3 :)
        smooth = 0
        t0 = self.joint_ctrl_history[0]
        t_minus1 = self.joint_ctrl_history[1]
        t_minus2 = self.joint_ctrl_history[2]

        for i in range(13):
            smooth += 2.5 * np.square(t0[i] - t_minus1[i]) + 1.5 * np.square(
                t0[i] - 2 * t_minus1[i] + t_minus2[i]
            )

        return -smooth

    def is_terminated(self) -> bool:
        rot = np.array(self.data.body("base").xmat).reshape(3, 3)
        Z_vec = rot[:, 2]
        upright = np.array([0, 0, 1])
        return (
            self.data.body("base").xpos[2] < 0.08 or np.dot(upright, Z_vec) <= 0
        )  # base z is below 0.08m or base has more than 90 degrees of tilt

    def follow_walk_reward(self):
        targets = self.sample_walk()
        reward = 0
        for joint_name, target in targets.items():
            joint_id = dofs[joint_name]
            joint_angle = self.data.qpos[7 + joint_id]
            reward += -np.square(joint_angle - target)
        return reward

    def step(self, a):
        # https://www.nature.com/articles/s41598-023-38259-7.pdf

        # add random force
        # if np.random.rand() < 0.05:
        #     self.data.xfrc_applied[self.data.body("base").id][:3] = [
        #         np.random.randint(-5, 5),
        #         np.random.randint(-5, 5),
        #         np.random.randint(-5, 5),
        #     ]  # absolute

        # angular distance to upright position in reward
        Z_vec = np.array(self.data.body("base").xmat).reshape(3, 3)[:, 2]
        upright_reward = np.square(np.dot(np.array([0, 0, 1]), Z_vec))

        walking_height_reward = (
            -np.square((self.get_body_com("base")[2] - 0.14)) * 100
        )  # "normal" walking height is about 0.14m

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
        # print(velocity_tracking_reward)

        smoothness_reward = self.compute_smoothness_reward()

        # add foot contact reward

        reward = (
            0.05  # time reward
            # + 0.1 * walking_height_reward
            + 0.1 * upright_reward
            + 2 * velocity_tracking_reward
            + 0.1 * smoothness_reward
            # + 0.1 * joint_angle_deviation_reward
        )

        self.do_simulation(a, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()

        # if self.is_terminated():
        # print(
        #     "Terminated because too much tilt or com too low.",
        # )
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
        # self.target_velocity = np.asarray([1, 0, 0])  # x, y, yaw
        self.target_velocity = np.asarray([2, 0, 0])  # x, y, yaw

        self.set_state(qpos, self.init_qvel)
        return self._get_obs()

    def goto_init(self):
        self.data.qpos[:] = self.init_qpos[:]
        for i, value in enumerate(init_pos.values()):
            self.data.qpos[i + 7] = value

    def _get_obs(self):

        joints_rotations = self.data.qpos[7 : 7 + 13]
        joints_velocities = self.data.qvel[6 : 6 + 13]

        joints_error = self.data.ctrl - self.data.qpos[7 : 7 + 13]
        self.joint_error_history.append(joints_error)
        self.joint_error_history = self.joint_error_history[
            -self.joint_history_length :
        ]

        angular_velocity = self.data.body("base").cvel[
            :3
        ]  # TODO this is imu, add noise to it later
        linear_velocity = self.data.body("base").cvel[3:]

        self.joint_ctrl_history.append(self.data.ctrl.copy())
        self.joint_ctrl_history = self.joint_ctrl_history[-self.joint_history_length :]

        return np.concatenate(
            [
                joints_rotations,
                joints_velocities,
                angular_velocity,
                linear_velocity,
                self.target_velocity,
                np.array(self.joint_error_history).flatten(),
                [np.sin(self.data.time)],
            ]
        )

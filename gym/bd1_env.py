import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


def calculate_signed_angle_xy(pose, target):
    """
    Calculate the signed angle between the y component of the rotation matrix and the vector from the origin to the target
    Args:
        pose: 4x4 rotation matrix
        target: 3D vector
    """
    y_component = pose[:3, 1]
    y_component_2d = np.array([y_component[0], y_component[1]])

    origin_to_target = np.array(target) - pose[:3, 3]
    origin_to_target_2d = np.array([origin_to_target[0], origin_to_target[1]])

    y_component_2d_normalized = y_component_2d / np.linalg.norm(y_component_2d)
    origin_to_target_2d_normalized = origin_to_target_2d / np.linalg.norm(
        origin_to_target_2d
    )
    angle_radians = np.arctan2(
        y_component_2d_normalized[0] * origin_to_target_2d_normalized[1]
        - y_component_2d_normalized[1] * origin_to_target_2d_normalized[0],
        np.dot(y_component_2d_normalized, origin_to_target_2d_normalized),
    )

    return angle_radians


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
    ### | 0    | Apply torque on right_hip_yaw                                     | -0.58       | 0.58        | right_hip_yaw                    | cylinder | torque (N m) |
    ### | 1    | Apply torque on right_hip_roll                                    | -0.58       | 0.58        | right_hip_roll                   | cylinder | torque (N m) |
    ### | 2    | Apply torque on right_hip_pitch                                   | -0.58       | 0.58        | right_hip_pitch                  | cylinder | torque (N m) |
    ### | 3    | Apply torque on right_knee_pitch                                  | -0.58       | 0.58        | right_knee_pitch                 | cylinder | torque (N m) |
    ### | 4    | Apply torque on right_ankle_pitch                                 | -0.58       | 0.58        | right_ankle_pitch                | cylinder | torque (N m) |
    ### | 5    | Apply torque on left_hip_yaw                                      | -0.58       | 0.58        | left_hip_yaw                     | cylinder | torque (N m) |
    ### | 6    | Apply torque on left_hip_roll                                     | -0.58       | 0.58        | left_hip_roll                    | cylinder | torque (N m) |
    ### | 7    | Apply torque on left_hip_pitch                                    | -0.58       | 0.58        | left_hip_pitch                   | cylinder | torque (N m) |
    ### | 8    | Apply torque on left_knee_pitch                                   | -0.58       | 0.58        | left_knee_pitch                  | cylinder | torque (N m) |
    ### | 9    | Apply torque on left_ankle_pitch                                  | -0.58       | 0.58        | left_ankle_pitch                 | cylinder | torque (N m) |

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
    | 23  | angle around z from base to target                       | -Inf | Inf |                                  |          | angle (rad)              |

    # | 23  | x component of 2D vector to the target                   | -Inf | Inf |                                  |          | position (m)             |
    # | 24  | y component of 2D vector to the target                   | -Inf | Inf |                                  |          | position (m)             |

    # TODO give angle around z axis to target
    # below for later
    | x13 | x position of the center of mass                         | -Inf | Inf |                                  |          | position (m)             |
    | x14 | y position of the center of mass                         | -Inf | Inf |                                  |          | position (m)             |
    | x15 | z position of the center of mass                         | -Inf | Inf |                                  |          | position (m)             |
    | x16 | roll angle of the base                                   | -Inf | Inf |                                  |          | angle (rad)              |
    | x17 | pitch angle of the base                                  | -Inf | Inf |                                  |          | angle (rad)              |
    | x18 | yaw angle of the base                                    | -Inf | Inf |                                  |          | angle (rad)              |
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, reached_goal_reward=20.0, **kwargs):
        utils.EzPickle.__init__(self, reached_goal_reward, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float64)
        self.goal_pos = np.asarray([0, 0, 0])
        self._reached_goal_reward = reached_goal_reward
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
        return (
            self.has_reached_goal() or np.dot(upright, Z_vec) <= 0
        )  # base has more than 90 degrees of tilt

    def has_reached_goal(self) -> bool:
        return bool(
            (
                np.linalg.norm(
                    self.get_body_com("base")[:2] - self.get_body_com("goal")[:2]
                )
                < 0.15
            )
        )

    def step(self, a):

        ctrl_reward = -np.square(a).sum()

        dist_reward = -np.linalg.norm(
            self.get_body_com("base")[:2] - self.get_body_com("goal")[:2]
        )

        # introduce angular distance to upright position in reward
        rot = np.array(self.data.body("base").xmat).reshape(3, 3)
        Z_vec = rot[:, 2]
        upright = np.array([0, 0, 1])
        upright_reward = np.square(np.dot(upright, Z_vec))

        walking_height_reward = (
            -np.square((self.get_body_com("base")[2] - 0.14)) * 100
        )  # "normal" walking height is about 0.14m

        angle_to_target_reward = -self.get_angle_to_target()

        goal_reward = self.has_reached_goal() * 1000

        # TODO try to add reward for being close to manually set init position ?
        reward = (
            walking_height_reward * 2
            + upright_reward
            # + ctrl_reward
            + dist_reward * 2
            + self.data.body("base").cvel[3:][1]  # y velocity
            + angle_to_target_reward
            + goal_reward
            + 0.05  # time reward
        )

        self.do_simulation(a, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()

        if self.is_terminated():
            print(
                "Terminated because",
                "has reached goal" if self.has_reached_goal() else "too much tilt",
            )
            # self.reset()  # not needed because autoreset is True in register

        return (
            ob,
            reward,
            self.is_terminated(),  # terminated
            False,  # truncated
            dict(
                walking_height_reward=walking_height_reward,
                upright_reward=upright_reward,
                ctrl_reward=ctrl_reward,
                dist_reward=dist_reward,
                angle_to_target_reward=angle_to_target_reward,
                goal_reward=goal_reward,
                time_reward=0.05,
            ),
        )

    def reset_model(self):
        # TODO maybe try to set the robot to a good starting position, knees bent etc
        qpos = self.init_qpos

        # goal pos is a random point a circle of radius 1 around the origin
        # angle = np.random.uniform(0, 2 * np.pi)
        # radius = 1  # m
        # x = radius * np.cos(angle)
        # y = radius * np.sin(angle)

        # self.goal_pos = np.asarray(
        #     [
        #         x,
        #         y,
        #         0.01,
        #     ]
        # )

        # Try with a fixed goal position first
        self.goal_pos = [0, -1, 0.01]

        qpos[-7:-4] = self.goal_pos

        self.set_state(qpos, self.init_qvel)
        return self._get_obs()

    def get_angle_to_target(self):
        base_pose = np.eye(4)
        base_pose[:3, :3] = np.array(self.data.body("base").xmat).reshape(3, 3)
        base_pose[:3, 3] = self.get_body_com("base")
        angle_to_target = calculate_signed_angle_xy(base_pose, self.goal_pos)
        return angle_to_target

    def _get_obs(self):
        # target_vec = self.get_body_com("goal")[:2] - self.get_body_com("base")[:2]

        rotations = self.data.qpos[8 : 8 + 10]
        velocities = self.data.qvel[8 : 8 + 10]

        # TODO This is the IMU, add noise to it when trying to go real
        rot = np.array(self.data.body("base").xmat).reshape(3, 3)
        Z_vec = rot[:, 2]
        return np.concatenate(
            [rotations, velocities, Z_vec, [self.get_angle_to_target()]]
        )

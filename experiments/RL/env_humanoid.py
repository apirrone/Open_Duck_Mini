# Mimicking the humanoidv4 reward

import numpy as np
import placo
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from mini_bdx.walk_engine import WalkEngine


def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()


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
    OBSOLETE

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
    | 26  | roll angular velocity                                    | -Inf | Inf |                                  |          |                          |
    | 27  | pitch angular velocity                                   | -Inf | Inf |                                  |          |                          |
    | 28  | yaw angular velocity                                     | -Inf | Inf |                                  |          |                          |
    | 29  | current x linear velocity                                | -Inf | Inf |                                  |          |                          |
    | 30  | current y linear velocity                                | -Inf | Inf |                                  |          |                          |
    | 31  | current z linear velocity                                | -Inf | Inf |                                  |          |                          |

    | 32  | left foot in contact with the floor                      | -Inf | Inf |                                  |          |                          |
    | 33  | right foot in contact with the floor                     | -Inf | Inf |                                  |          |                          |
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
        observation_space = Box(low=-np.inf, high=np.inf, shape=(41,), dtype=np.float64)

        self.left_foot_in_contact = 0
        self.right_foot_in_contact = 0

        self._healthy_z_range = (0.1, 0.2)

        self._forward_reward_weight = 1.25
        self._ctrl_cost_weight = 0.3
        self.healthy_reward = 5.0

        MujocoEnv.__init__(
            self,
            "/home/antoine/MISC/mini_BDX/mini_bdx/robots/bdx/scene.xml",
            5,  # frame_skip TODO set to 1 to test
            observation_space=observation_space,
            **kwargs,
        )

    def check_contact(self, body1_name, body2_name):
        body1_id = self.data.body(body1_name).id
        body2_id = self.data.body(body2_name).id

        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            if (
                self.model.geom_bodyid[contact.geom1] == body1_id
                and self.model.geom_bodyid[contact.geom2] == body2_id
            ) or (
                self.model.geom_bodyid[contact.geom1] == body2_id
                and self.model.geom_bodyid[contact.geom2] == body1_id
            ):
                return True

        return False

    def control_cost(self):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))
        return control_cost

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z

        return is_healthy

    @property
    def terminated(self):
        rot = np.array(self.data.body("base").xmat).reshape(3, 3)
        Z_vec = rot[:, 2]
        upright = np.array([0, 0, 1])
        return not self.is_healthy or np.dot(upright, Z_vec) <= 0

    def step(self, a):
        xy_position_before = mass_center(self.model, self.data)
        self.do_simulation(a, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        self.right_foot_in_contact = self.check_contact("foot_module", "floor")
        self.left_foot_in_contact = self.check_contact("foot_module_2", "floor")

        ctrl_cost = self.control_cost()
        moving_cost = 1.0 * (abs(x_velocity) + np.abs(x_velocity))

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        # rewards = forward_reward + healthy_reward
        rewards = healthy_reward
        reward = rewards - ctrl_cost - moving_cost

        # print("healthy", healthy_reward)
        # print("ctrl", ctrl_cost)
        # print("moving", moving_cost)
        # print(("total reward", reward))
        # print("-")

        ob = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return (ob, reward, self.terminated, False, {})

    def reset_model(self):
        noise_low = -1e-2
        noise_high = 1e-2

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def _get_obs(self):

        position = (
            self.data.qpos.flat.copy()
        )  # position/rotation of trunk + position of all joints
        velocity = (
            self.data.qvel.flat.copy()
        )  # positional/angular velocity of trunk +  of all joints

        # com_inertia = self.data.cinert.flat.copy()
        # com_velocity = self.data.cvel.flat.copy()

        # actuator_forces = self.data.qfrc_actuator.flat.copy()
        # external_contact_forces = self.data.cfrc_ext.flat.copy()

        obs = np.concatenate(
            [
                position,
                velocity,
                [self.left_foot_in_contact, self.right_foot_in_contact],
            ]
        )
        # print("OBS SIZE", len(obs))
        return obs

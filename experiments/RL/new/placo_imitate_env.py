import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from mini_bdx.placo_walk_engine import PlacoWalkEngine
from mini_bdx.utils.mujoco_utils import check_contact

# from placo_utils.visualization import robot_viz


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
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(101,), dtype=np.float64
        )
        self.target_velocity = np.asarray([0, 0, 0])  # x, y, yaw
        self.joint_history_length = 3
        self.joint_error_history = self.joint_history_length * [15 * [0]]
        self.joint_ctrl_history = self.joint_history_length * [15 * [0]]

        self.left_foot_in_contact = 0
        self.right_foot_in_contact = 0
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

        # self.viz = robot_viz(self.pwe.robot)
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

    def get_feet_contact(self):
        right_contact = check_contact(self.data, self.model, "foot_module", "floor")
        left_contact = check_contact(self.data, self.model, "foot_module_2", "floor")
        return right_contact, left_contact

    def follow_placo_reward(self, action):
        placo_angles = list(self.pwe.get_angles().values())
        # print(np.around(placo_angles, 2))
        error = np.linalg.norm(placo_angles - action)
        return -np.square(error)

    def step(self, a):

        t = self.data.time
        dt = t - self.prev_t
        if self.startup_cooldown > 0:
            self.startup_cooldown -= dt

        if self.startup_cooldown > 0:
            self.do_simulation(self.init_pos, FRAME_SKIP)
            reward = 0
        else:

            self.do_simulation(a, FRAME_SKIP)

            # self.right_foot_in_contact, self.left_foot_in_contact = (
            #     self.get_feet_contact()
            # )

            self.pwe.tick(
                dt
            )  # , self.left_foot_in_contact, self.right_foot_in_contact)

            reward = (
                0.05  # time reward
                + self.follow_placo_reward(a)
                # + 0.1 * self.walking_height_reward()
            )
            # print(self.follow_placo_reward(a))

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

        self.joint_error_history = self.joint_history_length * [15 * [0]]
        self.joint_ctrl_history = self.joint_history_length * [15 * [0]]

        self.target_velocity = np.asarray([0, 0, 0])  # x, y, yaw

        self.set_state(self.data.qpos, self.data.qvel)
        return self._get_obs()

    def goto_init(self):
        self.data.qvel[:] = np.zeros(len(self.data.qvel[:]))
        self.data.qpos[7 : 7 + 15] = self.init_pos
        self.data.qpos[2] = 0.15
        self.data.qpos[3 : 3 + 4] = [1, 0, 0.08, 0]

        self.data.ctrl[:] = self.init_pos

    def _get_obs(self):

        joints_rotations = self.data.qpos[7 : 7 + 15]
        joints_velocities = self.data.qvel[6 : 6 + 15]

        angular_velocity = self.data.body("base").cvel[
            :3
        ]  # TODO this is imu, add noise to it later
        linear_velocity = self.data.body("base").cvel[3:]

        self.joint_ctrl_history.append(self.data.ctrl.copy())
        self.joint_ctrl_history = self.joint_ctrl_history[-self.joint_history_length :]
        placo_angles = list(self.pwe.get_angles().values())
        return np.concatenate(
            [
                joints_rotations,
                joints_velocities,
                angular_velocity,
                linear_velocity,
                self.target_velocity,
                np.array(self.joint_ctrl_history).flatten(),
                [self.left_foot_in_contact, self.right_foot_in_contact],
                placo_angles,
            ]
        )

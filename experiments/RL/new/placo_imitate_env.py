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
        self.nb_dofs = 15

        self.target_velocity = np.asarray([0, 0, 0])  # x, y, yaw
        self.joint_history_length = 3
        self.joint_error_history = self.joint_history_length * [self.nb_dofs * [0]]
        self.joint_ctrl_history = self.joint_history_length * [self.nb_dofs * [0]]

        # observation_space = Box(
        #     low=-np.inf, high=np.inf, shape=(101,), dtype=np.float64
        # )

        observation_space = Box(
            np.array(
                [
                    *(-np.pi * np.ones(self.nb_dofs)),  # joints_rotations
                    *(-10 * np.ones(self.nb_dofs)),  # joints_velocities
                    *(-10 * np.ones(3)),  # angular_velocity
                    *(-10 * np.ones(3)),  # linear_velocity
                    *(-10 * np.ones(3)),  # target_velocity
                    *(
                        -np.pi * np.ones(self.nb_dofs * self.joint_history_length)
                    ),  # joint_ctrl_history
                    *(np.zeros(2)),  # feet_contact
                    *(-np.pi * np.ones(self.nb_dofs)),  # placo_angles
                ]
            ),
            np.array(
                [
                    *(np.pi * np.ones(self.nb_dofs)),  # joints_rotations
                    *(10 * np.ones(self.nb_dofs)),  # joints_velocities
                    *(10 * np.ones(3)),  # angular_velocity
                    *(10 * np.ones(3)),  # linear_velocity
                    *(10 * np.ones(3)),  # target_velocity
                    *(
                        np.pi * np.ones(self.nb_dofs * self.joint_history_length)
                    ),  # joint_ctrl_history
                    *(np.ones(2)),  # feet_contact
                    *(np.pi * np.ones(self.nb_dofs)),  # placo_angles
                ]
            ),
        )

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

    def follow_placo_reward(self):
        current_pos = self.data.qpos[7 : 7 + self.nb_dofs]
        placo_angles = list(self.pwe.get_angles().values())
        # print(np.around(placo_angles, 2))
        error = np.linalg.norm(placo_angles - current_pos)
        return -np.square(error)

    def walking_height_reward(self):
        return (
            -np.square((self.get_body_com("base")[2] - 0.14)) * 100
        )  # "normal" walking height is about 0.14m

    def velocity_tracking_reward(self):
        base_velocity = list(self.data.body("base").cvel[3:][:2]) + [
            self.data.body("base").cvel[:3][2]
        ]
        base_velocity = np.asarray(base_velocity)
        return np.exp(-np.square(base_velocity - self.target_velocity).sum())

    def upright_reward(self):
        # angular distance to upright position in reward
        Z_vec = np.array(self.data.body("base").xmat).reshape(3, 3)[:, 2]
        return np.square(np.dot(np.array([0, 0, 1]), Z_vec))

    def smoothness_reward2(self):
        # Warning, this function only works if the history is 3 :)
        smooth = 0
        t0 = self.joint_ctrl_history[0]
        t_minus1 = self.joint_ctrl_history[1]
        t_minus2 = self.joint_ctrl_history[2]

        for i in range(15):
            smooth += np.square(t0[i] - t_minus1[i]) + np.square(
                t_minus1[i] - t_minus2[i]
            )
            # smooth += 2.5 * np.square(t0[i] - t_minus1[i]) + 1.5 * np.square(
            #     t0[i] - 2 * t_minus1[i] + t_minus2[i]
            # )

        return -smooth

    def step(self, a):

        t = self.data.time
        dt = t - self.prev_t
        if self.startup_cooldown > 0:
            self.startup_cooldown -= dt

        if self.startup_cooldown > 0:
            self.do_simulation(self.init_pos, FRAME_SKIP)
            reward = 0
        else:

            current_ctrl = self.data.ctrl.copy()
            # Limiting the control
            delta_max = 0.1
            # print("a before clipping", a)
            a = np.clip(a, current_ctrl - delta_max, current_ctrl + delta_max)
            # print("a after clipping", a)

            self.do_simulation(a, FRAME_SKIP)

            # self.right_foot_in_contact, self.left_foot_in_contact = (
            #     self.get_feet_contact()
            # )

            self.pwe.tick(
                dt
            )  # , self.left_foot_in_contact, self.right_foot_in_contact)

            reward = (
                0.05  # time reward
                + 0.1 * self.walking_height_reward()
                + 0.1 * self.upright_reward()
                + 0.1 * self.velocity_tracking_reward()
                + 0.01 * self.smoothness_reward2()
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

        self.joint_error_history = self.joint_history_length * [self.nb_dofs * [0]]
        self.joint_ctrl_history = self.joint_history_length * [self.nb_dofs * [0]]

        self.target_velocity = np.asarray([0.2, 0, 0])  # x, y, yaw

        self.set_state(self.data.qpos, self.data.qvel)
        return self._get_obs()

    def goto_init(self):
        self.data.qvel[:] = np.zeros(len(self.data.qvel[:]))
        self.data.qpos[7 : 7 + self.nb_dofs] = self.init_pos
        self.data.qpos[2] = 0.15
        self.data.qpos[3 : 3 + 4] = [1, 0, 0.08, 0]

        self.data.ctrl[:] = self.init_pos

    def _get_obs(self):

        joints_rotations = self.data.qpos[7 : 7 + self.nb_dofs]
        joints_velocities = self.data.qvel[6 : 6 + self.nb_dofs]

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

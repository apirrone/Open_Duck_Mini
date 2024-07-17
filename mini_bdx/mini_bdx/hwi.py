import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from mini_bdx.io_330 import Dxl330IO


class HWI:
    def __init__(self, usb_port="/dev/ttyUSB1", baudrate=1000000):
        self.dxl_io = Dxl330IO(usb_port, baudrate=baudrate)
        self.joints = {
            "right_hip_yaw": 10,
            "right_hip_roll": 11,
            "right_hip_pitch": 12,
            "right_knee": 13,
            "right_ankle": 14,
            "left_hip_yaw": 20,
            "left_hip_roll": 21,
            "left_hip_pitch": 22,
            "left_knee": 23,
            "left_ankle": 24,
            "neck_pitch": 30,
            "head_pitch": 31,
            "head_yaw": 32,
        }
        self.init_pos = {
            "right_hip_yaw": 0.0012322806287681889,
            "right_hip_roll": 0.02326413299385176,
            "right_hip_pitch": 0.897352997720036,
            "right_knee": -1.6590427732988653,
            "right_ankle": 0.7617041101973798,
            "left_hip_yaw": -0.0012322806287510275,
            "left_hip_roll": 0.02326413299396169,
            "left_hip_pitch": 0.9488873968876821,
            "left_knee": -1.6490097909463939,
            "left_ankle": 0.7001367286772635,
            "neck_pitch": -0.1835609559422233,
            "head_pitch": -0.1834247585248765,
            "head_yaw": -9.174169188795582e-16,
        }

        # current based position
        self.dxl_io.set_operating_mode({id: 0x3 for id in self.joints.values()})

    def set_low_torque(self):
        self.dxl_io.set_pid_gain({id: [100, 0, 0] for id in self.joints.values()})

    def set_high_torque(self):
        self.dxl_io.set_pid_gain({id: [2000, 0, 5] for id in self.joints.values()})
        # TODO better with I and D ?
        # self.dxl_io.set_pid_gain(
        #     {id: [2500, 1000, 1000] for id in self.joints.values()}
        # )
        for name in ["neck_pitch", "head_pitch", "head_yaw"]:
            self.dxl_io.set_pid_gain({self.joints[name]: [150, 0, 0]})

    def turn_on(self):
        # self.set_low_torque()
        self.dxl_io.enable_torque(self.joints.values())
        self.set_position_all(self.init_pos)
        time.sleep(1)
        self.set_high_torque()

    def turn_off(self):
        self.dxl_io.disable_torque(self.joints.values())

    def goto_zero(self):
        goal = {joint: 0 for joint in self.joints.values()}
        self.dxl_io.set_goal_position(goal)

    # def goto_init(self):
    #     present_position = list(self.dxl_io.get_present_position(self.joints.values()))
    #     for i in range(len(present_position)):
    #         present_position[i] = np.deg2rad(present_position[i])
    #     print(present_position)

    #     init = {
    #         "right_hip_yaw": -0.0012322806287681889,
    #         "right_hip_roll": -0.02326413299385176,
    #         "right_hip_pitch": -0.897352997720036,
    #         "right_knee": 1.6590427732988653,
    #         "right_ankle": -0.7617041101973798,
    #         "left_hip_yaw": 0.0012322806287510275,
    #         "left_hip_roll": -0.02326413299396169,
    #         "left_hip_pitch": -0.9488873968876821,
    #         "left_knee": 1.6490097909463939,
    #         "left_ankle": -0.7001367286772635,
    #         "neck_pitch": 0.1835609559422233,
    #         "head_pitch": 0.1834247585248765,
    #         "head_yaw": 9.174169188795582e-16,
    #     }
    #     init_position = list(init.values())
    #     n_steps = 100
    #     interp_funcs = [
    #         interp1d([0, 1], [p, g]) for p, g in zip(present_position, init_position)
    #     ]
    #     interpolated_values = np.array(
    #         [[f(i / n_steps) for f in interp_funcs] for i in range(n_steps + 1)]
    #     )

    #     for values in interpolated_values:
    #         goal = {
    #             joint: np.rad2deg(position)
    #             for joint, position in zip(self.joints.keys(), values)
    #         }
    #         self.set_position_all(goal)
    #         time.sleep(0.1)

    def set_position_all(self, joints_positions):
        """
        joints_positions is a dictionary with joint names as keys and joint positions as values
        Warning: expects radians
        """
        ids_positions = {
            self.joints[joint]: np.rad2deg(-position)
            for joint, position in joints_positions.items()
        }

        # print(ids_positions)
        self.dxl_io.set_goal_position(ids_positions)

    def get_present_current(self, joint_name):
        return self.dxl_io.get_present_current([self.joints[joint_name]])[0]

    def get_goal_current(self, joint_name):
        return self.dxl_io.get_goal_current([self.joints[joint_name]])[0]

    def get_current_limit(self, joint_name):
        return self.dxl_io.get_current_limit([self.joints[joint_name]])[0]

    def get_present_positions(self):
        present_position = list(
            np.around(
                np.deg2rad((self.dxl_io.get_present_position(self.joints.values()))), 3
            )
        )
        factor = np.ones(len(present_position)) * -1
        return present_position * factor

    def get_operating_modes(self):
        return self.dxl_io.get_operating_mode(self.joints.values())

import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from mini_bdx.io_330 import Dxl330IO


class HWI:
    def __init__(self, usb_port="/dev/ttyUSB0", baudrate=1000000):
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
            "head_pitch1": 30,
            "head_pitch2": 31,
            "head_yaw": 32,
        }
        self.inverted_joints = [
            "right_hip_pitch",
            "left_hip_pitch",
            "right_knee",
            "left_knee",
            "right_ankle",
            "left_ankle",
        ]
        self.dxl_io.set_pid_gain({id: [1500, 0, 0] for id in self.joints.values()})

    def turn_on(self):
        self.dxl_io.enable_torque(self.joints.values())

    def turn_off(self):
        self.dxl_io.disable_torque(self.joints.values())

    def goto_zero(self):
        goal = {joint: 0 for joint in self.joints.values()}
        self.dxl_io.set_goal_position(goal)

    def goto_init(self):
        present_position = list(self.dxl_io.get_present_position(self.joints.values()))
        for i in range(len(present_position)):
            present_position[i] = np.deg2rad(present_position[i])
        print(present_position)

        init = {
            "right_hip_yaw": -0.0012322806287681889,
            "right_hip_roll": -0.02326413299385176,
            "right_hip_pitch": -0.897352997720036,
            "right_knee": 1.6590427732988653,
            "right_ankle": -0.7617041101973798,
            "left_hip_yaw": 0.0012322806287510275,
            "left_hip_roll": -0.02326413299396169,
            "left_hip_pitch": -0.9488873968876821,
            "left_knee": 1.6490097909463939,
            "left_ankle": -0.7001367286772635,
            "head_pitch1": 0.1835609559422233,
            "head_pitch2": 0.1834247585248765,
            "head_yaw": 9.174169188795582e-16,
        }
        init_position = list(init.values())
        n_steps = 100
        interp_funcs = [
            interp1d([0, 1], [p, g]) for p, g in zip(present_position, init_position)
        ]
        interpolated_values = np.array(
            [[f(i / n_steps) for f in interp_funcs] for i in range(n_steps + 1)]
        )

        for values in interpolated_values:
            goal = {
                joint: np.rad2deg(position)
                for joint, position in zip(self.joints.keys(), values)
            }
            # for joint in self.inverted_joints:
            #     goal[self.joints[joint]] *= -1
            self.set_position_all(goal)
            time.sleep(0.1)

    def set_position_all(self, joints_positions):
        """
        joints_positions is a dictionary with joint names as keys and joint positions as values
        Warning: expects radians
        """
        ids_positions = {
            self.joints[joint]: np.rad2deg(-position)
            for joint, position in joints_positions.items()
        }
        # for joint in self.inverted_joints:
        #     ids_positions[self.joints[joint]] *= -1

        # print(ids_positions)
        self.dxl_io.set_goal_position(ids_positions)

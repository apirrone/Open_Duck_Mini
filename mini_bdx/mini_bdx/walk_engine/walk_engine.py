# Based on https://github.com/Rhoban/walk_engine

import FramesViewer.utils as fv_utils
import numpy as np
import placo

from mini_bdx.utils import PolySpline


class FootPose:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.yaw = 0

    def __eq__(self, other):
        return (
            self.x == other.x
            and self.y == other.y
            and self.z == other.z
            and self.yaw == other.yaw
        )

    @property
    def foot_to_trunk(self):
        # Returns a frame from foot to trunk frame (this is actually a 2d matrix)
        # TODO
        pass


class Foot:
    def __init__(self):
        self.x_spline = PolySpline()
        self.y_spline = PolySpline()
        self.z_spline = PolySpline()
        self.yaw_spline = PolySpline()
        self.trunk_y_offset = 0

    def get_position(self, t):
        # Get the foot position, t is from 0 to 1, playing the footstep
        foot_pose = FootPose()
        foot_pose.x = self.x_spline.get(t)
        foot_pose.y = self.y_spline.get(t)
        foot_pose.z = self.z_spline.get(t)
        foot_pose.yaw = self.yaw_spline.get(t)
        return foot_pose

    def clear_splines(self):
        self.x_spline.clear()
        self.y_spline.clear()
        self.z_spline.clear()
        self.yaw_spline.clear()

    def copy(self):
        foot = Foot()
        foot.x_spline = self.x_spline.copy()
        foot.y_spline = self.y_spline.copy()
        foot.z_spline = self.z_spline.copy()
        foot.yaw_spline = self.yaw_spline.copy()
        foot.trunk_y_offset = self.trunk_y_offset
        return foot


class WalkEngine:
    def __init__(
        self,
        robot: placo.RobotWrapper,
        default_trunk_x_offset: float = 0.007,  # 0.007
        default_trunk_z_offset: float = -0.003,
        foot_y_offset: float = 0.0,
        max_rise_gain: float = 0.015,
        rise_duration: float = 0.2,
        frequency: float = 2.0,
        swing_gain: float = -0.001,
        swing_phase: float = 0.0,
        foot_y_offset_per_step_size_y: float = 0.02,
        target_trunk_pitch: float = 0,
        target_trunk_roll: float = 0,
        step_size_x: float = 0,
        step_size_y: float = 0,
        step_size_yaw: float = 0,
    ):
        kinematics_solver = placo.KinematicsSolver(robot)
        self.left = Foot()
        self.right = Foot()
        self.swing_spline = PolySpline()

        self.trunk_pitch = 0
        self.trunk_roll = 0

        self.trunk_pitch_timeout = 1.0  # s

        self.target_trunk_pitch = target_trunk_pitch
        self.target_trunk_roll = target_trunk_roll

        self.trunk_pitch_roll_compensation = False

        self.robot = robot
        self.kinematics_solver = kinematics_solver

        self.T_world_trunk = np.eye(4)
        self.T_world_trunk = fv_utils.rotateInSelf(
            self.T_world_trunk, [0, self.trunk_pitch, 0], degrees=True
        )

        self.T_world_head = robot.get_T_world_frame("head")
        self.T_world_head = fv_utils.translateInSelf(
            self.T_world_head, [-0.05, 0, -0.05]
        )
        self.head_task = self.kinematics_solver.add_frame_task(
            "head", self.T_world_head
        )
        self.head_task.configure("head", "soft")

        self.trunk_task = self.kinematics_solver.add_frame_task(
            "trunk", self.T_world_trunk
        )
        self.trunk_task.configure("trunk", "hard")

        self.right_foot_task = self.kinematics_solver.add_frame_task(
            "right_foot", robot.get_T_world_frame("right_foot")
        )
        self.right_foot_task.configure("right_foot", "soft", 5.0, 0.1)

        self.left_foot_task = self.kinematics_solver.add_frame_task(
            "left_foot", robot.get_T_world_frame("left_foot")
        )
        self.left_foot_task.configure("left_foot", "soft", 5.0, 0.1)

        self.is_left_support = False

        self.trunk_height = -robot.get_T_world_frame("left_foot")[:3, 3][2] / 1.2
        self.foot_distance = np.abs(robot.get_T_world_frame("left_foot")[:3, 3][1])

        self.default_trunk_x_offset = default_trunk_x_offset
        self.forward_trunk_x_offset = self.default_trunk_x_offset - 0.002
        self.backward_trunk_x_offset = self.default_trunk_x_offset
        self.tune_trunk_x_offset = 0

        self.default_trunk_z_offset = default_trunk_z_offset
        self.foot_y_offset = foot_y_offset
        self.max_rise_gain = max_rise_gain
        self.rise_gain = 0
        self.rise_duration = rise_duration
        self.frequency = frequency
        self.swing_gain = swing_gain
        self.swing_phase = swing_phase
        self.foot_y_offset_per_step_size_y = foot_y_offset_per_step_size_y
        self.step_size_x = step_size_x
        self.step_size_y = step_size_y
        self.step_size_yaw = step_size_yaw

        self.time_since_last_step = 0
        self.time_since_last_left_contact = 0
        self.time_since_last_right_contact = 0

        self.step_duration = 0
        self._swing_gain = 0

        self.reset()

    def get_left_foot_pose(self, t):
        left_position = self.left.get_position(t)

        T_world_left_foot = np.eye(4)
        T_world_left_foot[:3, 3] = [
            left_position.x,
            left_position.y,
            left_position.z,
        ]
        T_world_left_foot = fv_utils.rotateInSelf(
            T_world_left_foot, [0, 0, left_position.yaw], degrees=False
        )

        return T_world_left_foot

    def get_right_foot_pose(self, time_since_last_step):
        right_position = self.right.get_position(time_since_last_step)
        T_world_right_foot = np.eye(4)
        T_world_right_foot[:3, 3] = [
            right_position.x,
            right_position.y,
            right_position.z,
        ]
        T_world_right_foot = fv_utils.rotateInSelf(
            T_world_right_foot, [0, 0, right_position.yaw], degrees=False
        )
        return T_world_right_foot

    # gyro is angular position of the trunk [roll, pitch, yaw]
    # accelerometer is the acceleration of the trunk [x, y, z]
    def update(
        self,
        walking,
        gyro,
        accelerometer,
        left_contact,
        right_contact,
        target_step_x,
        target_step_y,
        target_yaw,
        target_head_pitch,
        target_head_yaw,
        target_head_z_offset,
        dt,
        ignore_feet_contact=False,
    ):
        if self.trunk_pitch_timeout > 0:
            self.trunk_pitch_timeout -= dt
        if left_contact:
            self.time_since_last_left_contact = 0
        if right_contact:
            self.time_since_last_right_contact = 0

        if ignore_feet_contact or (
            not self.time_since_last_left_contact > self.rise_duration
            and not self.time_since_last_right_contact > self.rise_duration
        ):
            self.time_since_last_step += dt

        self.time_since_last_left_contact += dt
        self.time_since_last_right_contact += dt

        if self.time_since_last_step > self.step_duration:
            self.time_since_last_step = 0
            self.new_step()

        target_rise_gain = self.max_rise_gain if walking else 0

        # slowly increase self.step_size_x and self.step_size_y to target_step_x and target_step_y
        # target can be negative or positive or 0
        delta_x = target_step_x - self.step_size_x
        delta_y = target_step_y - self.step_size_y
        delta_yaw = target_yaw - self.step_size_yaw
        delta_rise_gain = target_rise_gain - self.rise_gain

        self.step_size_x = self.step_size_x + (delta_x / 100)
        self.step_size_y = self.step_size_y + (delta_y / 100)
        self.step_size_yaw = self.step_size_yaw + (delta_yaw / 100)
        self.rise_gain = self.rise_gain + (delta_rise_gain / 1000)

        swing = 0
        if walking:
            self.left_foot_task.T_world_frame = self.get_left_foot_pose(
                self.time_since_last_step
            )
            self.right_foot_task.T_world_frame = self.get_right_foot_pose(
                self.time_since_last_step
            )

            swing_P = 0 if self.is_left_support else np.pi
            swing_P += np.pi * 2 * self.swing_phase
            swing = self._swing_gain * np.sin(
                np.pi * self.time_since_last_step / self.step_duration + swing_P
            )
        else:
            self.step_size_x = 0
            self.step_size_y = 0
            self.step_size_yaw = 0
            self.left_foot_task.T_world_frame = self.get_left_foot_pose(0)
            self.right_foot_task.T_world_frame = self.get_right_foot_pose(0)

        # Trunk pitch and roll
        if self.trunk_pitch_roll_compensation:
            self.trunk_pitch = max(-30, min(30, -np.rad2deg(gyro[1]) * 5))
            self.trunk_roll = max(-10, min(10, np.rad2deg(gyro[0])))
        # else:
        #     self.trunk_pitch = 0
        #     self.trunk_roll = 0

        if self.trunk_pitch_timeout <= 0:
            delta_trunk_pitch = self.target_trunk_pitch - self.trunk_pitch
            self.trunk_pitch = self.trunk_pitch + (delta_trunk_pitch / 100)

        self.T_world_trunk = np.eye(4)
        self.T_world_trunk = fv_utils.rotateInSelf(
            self.T_world_trunk, [self.trunk_roll, self.trunk_pitch, 0], degrees=True
        )
        self.T_world_trunk[:3, 3] = [0, swing, 0]

        fr = self.T_world_trunk
        fr[:3, 3] = [0, swing, 0]
        self.trunk_task.T_world_frame = fr

        # Head
        tmp = self.T_world_head.copy()
        tmp = fv_utils.translateInSelf(tmp, [0, 0, -target_head_z_offset])
        tmp = fv_utils.rotateInSelf(
            tmp, [0, target_head_pitch, target_head_yaw], degrees=False
        )
        self.head_task.T_world_frame = tmp

        self.robot.update_kinematics()
        self.kinematics_solver.solve(True)

    def get_angles(self):
        angles = {
            "right_hip_yaw": self.robot.get_joint("right_hip_yaw"),
            "right_hip_roll": self.robot.get_joint("right_hip_roll"),
            "right_hip_pitch": self.robot.get_joint("right_hip_pitch"),
            "right_knee": self.robot.get_joint("right_knee"),
            "right_ankle": self.robot.get_joint("right_ankle"),
            "left_hip_yaw": self.robot.get_joint("left_hip_yaw"),
            "left_hip_roll": self.robot.get_joint("left_hip_roll"),
            "left_hip_pitch": self.robot.get_joint("left_hip_pitch"),
            "left_knee": self.robot.get_joint("left_knee"),
            "left_ankle": self.robot.get_joint("left_ankle"),
            "neck_pitch": self.robot.get_joint("neck_pitch"),
            "head_pitch": self.robot.get_joint("head_pitch"),
            "head_yaw": self.robot.get_joint("head_yaw"),
        }
        return angles

    def reset(self):
        self.left.trunk_y_offset = self.foot_distance + self.foot_y_offset
        self.right.trunk_y_offset = -(self.foot_distance + self.foot_y_offset)

        self.is_left_support = False

        self.step_duration = 1.0 / (2 * self.frequency)
        self.left.clear_splines()
        self.right.clear_splines()
        self.left.x_spline.add_point(self.step_duration, self.trunk_x_offset, 0)
        self.left.y_spline.add_point(self.step_duration, self.left.trunk_y_offset, 0)
        self.left.yaw_spline.add_point(self.step_duration, 0, 0)
        self.right.x_spline.add_point(self.step_duration, self.trunk_x_offset, 0)
        self.right.y_spline.add_point(self.step_duration, self.right.trunk_y_offset, 0)
        self.right.yaw_spline.add_point(self.step_duration, 0, 0)

        self.trunk_pitch = 0
        self.trunk_roll = 0

        self.new_step()

    def new_step(self):
        self._swing_gain = self.swing_gain
        previous_step_duration = self.step_duration
        self.step_duration = 1.0 / (2 * self.frequency)

        old_left = self.left.copy()
        old_right = self.right.copy()
        self.left.clear_splines()
        self.right.clear_splines()

        self.left.trunk_y_offset = (
            self.foot_distance
            + self.foot_y_offset
            + self.foot_y_offset_per_step_size_y * np.abs(self.step_size_y)
        )
        self.right.trunk_y_offset = -(
            self.foot_distance
            + self.foot_y_offset
            + self.foot_y_offset_per_step_size_y * np.abs(self.step_size_y)
        )

        self.left.x_spline.add_point(
            0,
            old_left.x_spline.get(previous_step_duration),
            old_left.x_spline.get_vel(previous_step_duration),
        )
        self.left.y_spline.add_point(
            0,
            old_left.y_spline.get(previous_step_duration),
            old_left.y_spline.get_vel(previous_step_duration),
        )
        self.left.yaw_spline.add_point(
            0,
            old_left.yaw_spline.get(previous_step_duration),
            old_left.yaw_spline.get_vel(previous_step_duration),
        )

        self.right.x_spline.add_point(
            0,
            old_right.x_spline.get(previous_step_duration),
            old_right.x_spline.get_vel(previous_step_duration),
        )
        self.right.y_spline.add_point(
            0,
            old_right.y_spline.get(previous_step_duration),
            old_right.y_spline.get_vel(previous_step_duration),
        )
        self.right.yaw_spline.add_point(
            0,
            old_right.yaw_spline.get(previous_step_duration),
            old_right.yaw_spline.get_vel(previous_step_duration),
        )

        self.is_left_support = not self.is_left_support

        step_low = -self.trunk_height + self.default_trunk_z_offset
        step_high = -self.trunk_height + self.default_trunk_z_offset + self.rise_gain
        self.support_foot.z_spline.add_point(0, step_low, 0)
        self.support_foot.z_spline.add_point(self.step_duration, step_low, 0)

        self.flying_foot.z_spline.add_point(0, step_low, 0)
        if self.rise_duration > 0:
            self.flying_foot.z_spline.add_point(
                self.step_duration * (0.5 - self.rise_duration / 2.0),
                step_high,
                0,
            )
            self.flying_foot.z_spline.add_point(
                self.step_duration * (0.5 + self.rise_duration / 2.0),
                step_high,
                0,
            )
        else:
            self.flying_foot.z_spline.add_point(self.step_duration * 0.5, step_high, 0)
        self.flying_foot.z_spline.add_point(self.step_duration, step_low, 0)

        self.plan_step_end()

    def plan_step_end(self):
        self.support_foot.x_spline.add_point(
            self.step_duration, self.trunk_x_offset - self.step_size_x / 2.0, 0
        )
        self.support_foot.y_spline.add_point(
            self.step_duration,
            self.support_foot.trunk_y_offset - self.step_size_y / 2.0,
            0,
        )

        self.flying_foot.x_spline.add_point(
            self.step_duration, self.trunk_x_offset + self.step_size_x / 2.0, 0
        )
        self.flying_foot.y_spline.add_point(
            self.step_duration,
            self.flying_foot.trunk_y_offset + self.step_size_y / 2.0,
            0,
        )

        self.support_foot.yaw_spline.add_point(
            self.step_duration, -self.step_size_yaw / 2, 0
        )
        self.flying_foot.yaw_spline.add_point(
            self.step_duration, self.step_size_yaw / 2, 0
        )

    def replan(self):
        splines = [
            self.support_foot.x_spline,
            self.flying_foot.x_spline,
            self.support_foot.y_spline,
            self.flying_foot.y_spline,
            self.support_foot.yaw_spline,
            self.flying_foot.yaw_spline,
        ]

        for spline in splines:
            old_spline = spline.copy()
            spline.clear()
            spline.add_point(0, old_spline.get(0), old_spline.get_vel(0))
            spline.add_point(
                self.time_since_last_step,
                old_spline.get(self.time_since_last_step),
                old_spline.get_vel(self.time_since_last_step),
            )

        self.plan_step_end()

    @property
    def support_foot(self):
        return self.left if self.is_left_support else self.right

    @property
    def flying_foot(self):
        return self.right if self.is_left_support else self.left

    @property
    def trunk_x_offset(self):
        if self.step_size_x > 0:
            return self.forward_trunk_x_offset + self.tune_trunk_x_offset
        elif self.step_size_x == 0:
            return self.default_trunk_x_offset + self.tune_trunk_x_offset
        else:
            return self.backward_trunk_x_offset + self.tune_trunk_x_offset

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
        kinematics_solver: placo.KinematicsSolver,
        trunk_x_offset: float = 0,
        trunk_z_offset: float = 0.02,
        foot_y_offset: float = 0.03,
        rise_gain: float = 0.04,
        rise_duration: float = 0.2,
        frequency: float = 1.5,
        swing_gain: float = 0.02,
        swing_phase: float = 0.0,
        foot_y_offset_per_step_size_y: float = 0.02,
        trunk_pitch: float = 0,
        step_size_x: float = 0,
        step_size_y: float = 0,
        step_size_yaw: float = 0,
    ):
        self.left = Foot()
        self.right = Foot()
        self.swing_spline = PolySpline()

        self.trunk_pitch = trunk_pitch

        self.robot = robot
        self.kinematics_solver = kinematics_solver

        # self.kinematics_solver.mask_fbase(True)
        self.T_world_trunk = np.eye(4)
        self.T_world_trunk = fv_utils.rotateInSelf(
            self.T_world_trunk, [0, self.trunk_pitch, 0], degrees=True
        )

        self.trunk_task = self.kinematics_solver.add_frame_task(
            "trunk", self.T_world_trunk
        )
        self.trunk_task.configure("trunk", "hard")

        self.right_foot_tip_task = self.kinematics_solver.add_frame_task(
            "right_foot_tip", robot.get_T_world_frame("right_foot_tip")
        )
        self.right_foot_tip_task.configure("right_foot_tip", "soft", 5.0, 0.1)

        self.left_foot_tip_task = self.kinematics_solver.add_frame_task(
            "left_foot_tip", robot.get_T_world_frame("left_foot_tip")
        )
        self.left_foot_tip_task.configure("left_foot_tip", "soft", 5.0, 0.1)

        self.is_left_support = False

        self.trunk_height = -robot.get_T_world_frame("left_foot_tip")[:3, 3][2] / 1.3
        self.foot_distance = np.abs(robot.get_T_world_frame("left_foot_tip")[:3, 3][1])

        self.trunk_x_offset = trunk_x_offset
        self.trunk_z_offset = trunk_z_offset
        self.foot_y_offset = foot_y_offset
        self.rise_gain = rise_gain
        self.rise_duration = rise_duration
        self.frequency = frequency
        self.swing_gain = swing_gain
        self.swing_phase = swing_phase
        self.foot_y_offset_per_step_size_y = foot_y_offset_per_step_size_y
        self.step_size_x = step_size_x
        self.step_size_y = step_size_y
        self.step_size_yaw = step_size_yaw

        self.step_duration = 0
        self._swing_gain = 0

        self.reset()

    def get_left_foot_pose(self, time_since_last_step):
        left_position = self.left.get_position(time_since_last_step)

        T_world_left_foot_tip = np.eye(4)
        T_world_left_foot_tip[:3, 3] = [
            left_position.x,
            left_position.y,
            left_position.z,
        ]
        T_world_left_foot_tip = fv_utils.rotateInSelf(
            T_world_left_foot_tip, [0, 0, left_position.yaw], degrees=False
        )

        return T_world_left_foot_tip

    def get_right_foot_pose(self, time_since_last_step):
        right_position = self.right.get_position(time_since_last_step)
        T_world_right_foot_tip = np.eye(4)
        T_world_right_foot_tip[:3, 3] = [
            right_position.x,
            right_position.y,
            right_position.z,
        ]
        T_world_right_foot_tip = fv_utils.rotateInSelf(
            T_world_right_foot_tip, [0, 0, right_position.yaw], degrees=False
        )
        return T_world_right_foot_tip

    # imu is angular position of the trunk [roll, pitch, yaw]
    def update(self, walking, imu, time_since_last_step):
        if time_since_last_step < 0:
            time_since_last_step = 0
        if time_since_last_step > self.step_duration:
            time_since_last_step = self.step_duration

        swing = 0
        if walking:
            self.left_foot_tip_task.T_world_frame = self.get_left_foot_pose(
                time_since_last_step
            )
            self.right_foot_tip_task.T_world_frame = self.get_right_foot_pose(
                time_since_last_step
            )

            swing_P = 0 if self.is_left_support else np.pi
            swing_P += np.pi * 2 * self.swing_phase
            swing = self._swing_gain * np.sin(
                np.pi * time_since_last_step / self.step_duration + swing_P
            )
        else:
            self.left_foot_tip_task.T_world_frame = self.get_left_foot_pose(0)
            self.right_foot_tip_task.T_world_frame = self.get_right_foot_pose(0)

        self.trunk_pitch = -np.rad2deg(imu[1]) * 2
        print(self.trunk_pitch)
        self.T_world_trunk = np.eye(4)
        self.T_world_trunk = fv_utils.rotateInSelf(
            self.T_world_trunk, [0, self.trunk_pitch, 0], degrees=True
        )
        self.T_world_trunk[:3, 3] = [0, swing, 0]

        fr = self.T_world_trunk
        fr[:3, 3] = [0, swing, 0]
        self.trunk_task.T_world_frame = fr

    def compute_angles(self):
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
            "head_pitch1": self.robot.get_joint("head_pitch1"),
            "head_pitch2": self.robot.get_joint("head_pitch2"),
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

        step_low = -self.trunk_height
        step_high = (
            -self.trunk_height
            + self.rise_gain
            # if (self.step_size_x or self.step_size_y)
            # else -self.trunk_height
        )
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

    def replan(self, time_since_last_step):
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
                time_since_last_step,
                old_spline.get(time_since_last_step),
                old_spline.get_vel(time_since_last_step),
            )

        self.plan_step_end()

    @property
    def support_foot(self):
        return self.left if self.is_left_support else self.right

    @property
    def flying_foot(self):
        return self.right if self.is_left_support else self.left

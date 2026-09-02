"""Canonical joint tables for the Open Duck Mini v2 robot.

Joint names and limits are taken from ``mini_bdx/robots/open_duck_mini_v2/robot.urdf``
(the single source of truth for the 16 revolute joints). If the URDF changes,
update the tables below to match.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

# ----------------------------------------------------------------------------
# Joint groups, in canonical (URDF) order.
# ----------------------------------------------------------------------------

LEG_JOINTS: tuple[str, ...] = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)

HEAD_JOINTS: tuple[str, ...] = (
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
)

ANTENNA_JOINTS: tuple[str, ...] = (
    "left_antenna",
    "right_antenna",
)

ALL_JOINTS: tuple[str, ...] = LEG_JOINTS + HEAD_JOINTS + ANTENNA_JOINTS

NUM_JOINTS: int = len(ALL_JOINTS)  # 16

JOINT_GROUPS: dict[str, tuple[str, ...]] = {
    "legs": LEG_JOINTS,
    "head": HEAD_JOINTS,
    "antennas": ANTENNA_JOINTS,
}

# name -> index into ALL_JOINTS
JOINT_INDEX: dict[str, int] = {name: i for i, name in enumerate(ALL_JOINTS)}

# ----------------------------------------------------------------------------
# Limits, parsed verbatim from mini_bdx/robots/open_duck_mini_v2/robot.urdf.
# Each revolute joint declares: <limit effort="1" velocity="20" lower=... upper=.../>
# Angles are radians, velocities rad/s.
# ----------------------------------------------------------------------------

# name -> (lower, upper) radians
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    # LEGS
    "left_ankle": (-1.5707963267948966, 1.5707963267948966),
    "left_knee": (-1.5707963267948966, 1.5707963267948966),
    "left_hip_pitch": (-1.2217304763960306, 0.5235987755982988),
    "left_hip_roll": (-0.4363323129985824, 0.4363323129985824),
    "left_hip_yaw": (-0.5235987755982988, 0.5235987755982988),
    "right_ankle": (-1.5707963267948966, 1.5707963267948966),
    "right_knee": (-1.5707963267948966, 1.5707963267948966),
    "right_hip_pitch": (-0.5235987755982988, 1.2217304763960306),
    "right_hip_roll": (-0.4363323129985824, 0.4363323129985824),
    "right_hip_yaw": (-0.5235987755982988, 0.5235987755982988),
    # ANTENNAS
    "left_antenna": (-1.5707963267948966, 1.5707963267948966),
    "right_antenna": (-1.5707963267948966, 1.5707963267948966),
    # HEAD
    "head_roll": (-0.5235987755982988, 0.5235987755982988),
    "head_yaw": (-2.792526803190927, 2.792526803190927),
    "head_pitch": (-0.7853981633974483, 0.7853981633974483),
    "neck_pitch": (-0.3490658503988659, 1.1344640137963142),
}

# name -> max velocity in rad/s (all joints declare velocity="20" in the URDF)
JOINT_VELOCITY_LIMITS: dict[str, float] = {
    name: 20.0 for name in ALL_JOINTS
}


def group_of(joint: str) -> str:
    """Return the group name ('legs' | 'head' | 'antennas') a joint belongs to."""
    for group, joints in JOINT_GROUPS.items():
        if joint in joints:
            return group
    raise KeyError(f"Unknown joint: {joint!r}")


def to_array(values: Mapping[str, float]) -> np.ndarray:
    """Convert a name->angle mapping into a full ordered 16-joint array.

    Joints not present in ``values`` default to 0.0.
    """
    arr = np.zeros(NUM_JOINTS, dtype=np.float32)
    for name, value in values.items():
        if name not in JOINT_INDEX:
            raise KeyError(f"Unknown joint: {name!r}")
        arr[JOINT_INDEX[name]] = value
    return arr


def to_dict(array: np.ndarray) -> dict[str, float]:
    """Convert a full ordered 16-joint array into a name->angle dict."""
    array = np.asarray(array).ravel()
    if array.shape[0] != NUM_JOINTS:
        raise ValueError(
            f"Expected {NUM_JOINTS} values, got shape {array.shape}"
        )
    return {name: float(array[JOINT_INDEX[name]]) for name in ALL_JOINTS}

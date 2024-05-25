import gymnasium as gym
import imitation
import numpy as np
import placo
from gymnasium.envs.registration import register
from gymnasium.spaces import Box
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_seeds, make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from mini_bdx.walk_engine import WalkEngine

robot = placo.RobotWrapper(
    "../../mini_bdx/robots/bdx/robot.urdf", placo.Flags.ignore_collisions
)
walk_engine = WalkEngine(robot)

# Check this out https://imitation.readthedocs.io/en/latest/algorithms/bc.html


def expert(obs, states, dones):
    global walk_engine
    dt = 1 / 60
    walk_engine.update(
        True,
        [0, 0, 0],
        [0, 0, 0],
        False,
        False,
        0.03,
        0,
        0,
        0,
        0,
        0,
        dt,
        ignore_feet_contact=True,
    )
    angles = list(walk_engine.get_angles().values())
    return angles, states

import pickle

import gymnasium as gym
import imitation.data.types
import numpy as np
from gymnasium.envs.registration import register
from gymnasium.spaces import Box
from imitation.algorithms import bc
from stable_baselines3.common.evaluation import evaluate_policy

# Check this out https://imitation.readthedocs.io/en/latest/algorithms/bc.html

dataset = pickle.load(open("dataset.pkl", "rb"))

observation_space = Box(low=-np.inf, high=np.inf, shape=(76,), dtype=np.float64)
action_space = Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float64)

rng = np.random.default_rng(0)

bc_trainer = bc.BC(
    observation_space=observation_space,
    action_space=action_space,
    demonstrations=dataset,
    rng=rng,
    device="cpu",
)
bc_trainer.train(n_epochs=10)


register(
    id="BDX_env",
    entry_point="env:BDXEnv",
    max_episode_steps=500,
    autoreset=True,
)

env = gym.make("BDX_env", render_mode=None)

reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
print(reward)

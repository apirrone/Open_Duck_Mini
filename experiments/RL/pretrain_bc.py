import argparse
import pickle

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from imitation.algorithms import bc
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Check this out https://imitation.readthedocs.io/en/latest/algorithms/bc.html

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True)
args = parser.parse_args()


dataset = pickle.load(open(args.dataset, "rb"))

register(id="BDX_env", entry_point="env:BDXEnv")

env = gym.make("BDX_env", render_mode=None)

rng = np.random.default_rng(0)

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=dataset,
    rng=rng,
    device="cpu",
    policy=PPO("MlpPolicy", env).policy,  # not working with SAC for some reason
)
bc_trainer.train(n_epochs=50)

bc_trainer.policy.save("bc_policy_ppo.zip")

# reward, _ = evaluate_policy(bc_trainer.policy, env, 1)
# print(reward)

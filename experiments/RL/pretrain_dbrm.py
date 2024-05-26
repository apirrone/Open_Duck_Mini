import argparse
import pickle
import pprint

import numpy as np
from gymnasium.envs.registration import register
from imitation.algorithms import density as db
from imitation.data import serialize
from imitation.util import util
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True)
args = parser.parse_args()

rng = np.random.default_rng(0)

register(id="BDX_env", entry_point="env:BDXEnv")
env = util.make_vec_env("BDX_env", rng=rng, n_envs=2)

dataset = pickle.load(open(args.dataset, "rb"))

imitation_trainer = PPO(
    ActorCriticPolicy, env, learning_rate=3e-4, gamma=0.95, ent_coef=1e-4, n_steps=2048
)
density_trainer = db.DensityAlgorithm(
    venv=env,
    rng=rng,
    demonstrations=dataset,
    rl_algo=imitation_trainer,
    density_type=db.DensityType.STATE_ACTION_DENSITY,
    is_stationary=True,
    kernel="gaussian",
    kernel_bandwidth=0.4,
    standardise_inputs=True,
    allow_variable_horizon=True,
)
density_trainer.train()


def print_stats(density_trainer, n_trajectories):
    stats = density_trainer.test_policy(n_trajectories=n_trajectories)
    print("True reward function stats:")
    pprint.pprint(stats)
    stats_im = density_trainer.test_policy(
        true_reward=False, n_trajectories=n_trajectories
    )
    print("Imitation reward function stats:")
    pprint.pprint(stats_im)


print("Stats before training:")
print_stats(density_trainer, 1)

density_trainer.train_policy(
    1000000,
    progress_bar=True,
)  # Train for 1_000_000 steps to approach expert performance.

print("Stats after training:")
print_stats(density_trainer, 1)

density_trainer.policy.save("density_policy_ppo.zip")

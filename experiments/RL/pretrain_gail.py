import argparse
import pickle

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

# Check this out https://imitation.readthedocs.io/en/latest/algorithms/bc.html

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True)
args = parser.parse_args()


dataset = pickle.load(open(args.dataset, "rb"))

register(id="BDX_env", entry_point="env:BDXEnv")

SEED = 42
rng = np.random.default_rng(SEED)
# env = gym.make("BDX_env", render_mode=None)
env = make_vec_env(
    "BDX_env",
    rng=rng,
    n_envs=8,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
)


learner = PPO(
    env=env,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0004,
    gamma=0.95,
    n_epochs=5,
    seed=SEED,
    tensorboard_log="logs",
)
reward_net = BasicRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
)
gail_trainer = GAIL(
    demonstrations=dataset,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=8,
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
    allow_variable_horizon=True,
)

print("evaluate the learner before training")
env.seed(SEED)
learner_rewards_before_training, _ = evaluate_policy(
    learner,
    env,
    100,
    return_episode_rewards=True,
)

print("train the learner and evaluate again")
gail_trainer.train(500000)  # Train for 800_000 steps to match expert.

env.seed(SEED)
learner_rewards_after_training, _ = evaluate_policy(
    learner,
    env,
    100,
    return_episode_rewards=True,
)

print("mean episode reward before training:", np.mean(learner_rewards_before_training))
print("mean episode reward after training:", np.mean(learner_rewards_after_training))

print("Save the policy")
gail_trainer.policy.save("gail_policy_ppo.zip")

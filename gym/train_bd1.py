import argparse
import os
from datetime import datetime

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from sb3_contrib import TQC
from stable_baselines3 import A2C, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise


def train(env, sb3_algo, model_dir, log_dir, pretrained=None, device="cuda"):
    n_actions = env.action_space.shape[-1]
    # SAC parameters found here https://github.com/hill-a/stable-baselines/issues/840#issuecomment-623171534
    if pretrained is None:
        match sb3_algo:
            case "SAC":
                model = SAC(
                    "MlpPolicy",
                    env,
                    verbose=1,
                    device=device,
                    tensorboard_log=log_dir,
                    action_noise=NormalActionNoise(
                        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
                    ),
                    # learning_starts=10000,
                    # batch_size=100,
                    # learning_rate=1e-3,
                    # train_freq=1000,
                    # gradient_steps=1000,
                    policy_kwargs=dict(net_arch=[400, 300]),
                )
            case "TD3":
                model = TD3(
                    "MlpPolicy", env, verbose=1, device=device, tensorboard_log=log_dir
                )
            case "A2C":
                model = A2C(
                    "MlpPolicy", env, verbose=1, device=device, tensorboard_log=log_dir
                )
            case "TQC":
                model = TQC(
                    "MlpPolicy", env, verbose=1, device=device, tensorboard_log=log_dir
                )
            case _:
                print("Algorithm not found")
                return
    else:
        match sb3_algo:
            case "SAC":
                model = SAC.load(
                    pretrained,
                    env=env,
                    verbose=1,
                    device="cuda",
                    tensorboard_log=log_dir,
                )
            case "TD3":
                model = TD3.load(
                    pretrained,
                    env=env,
                    verbose=1,
                    device="cuda",
                    tensorboard_log=log_dir,
                )
            case "A2C":
                model = A2C.load(
                    pretrained,
                    env=env,
                    verbose=1,
                    device="cuda",
                    tensorboard_log=log_dir,
                )
            case "TQC":
                model = TQC.load(
                    pretrained,
                    env=env,
                    verbose=1,
                    device="cuda",
                    tensorboard_log=log_dir,
                )
            case _:
                print("Algorithm not found")
                return

    TIMESTEPS = 10000
    iters = 0
    while True:
        iters += 1

        model.learn(
            total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            progress_bar=True,
        )
        model.save(f"{model_dir}/{sb3_algo}_{TIMESTEPS*iters}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train bd1")
    parser.add_argument(
        "-a", "--algo", type=str, choices=["SAC", "TD3", "A2C", "TQC"], default="SAC"
    )
    parser.add_argument("-p", "--pretrained", type=str, required=False)
    parser.add_argument("-d", "--device", type=str, required=False, default="cuda")

    parser.add_argument(
        "-n",
        "--name",
        type=str,
        required=False,
        default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        help="Name of the experiment",
    )

    register(
        id="BD1_env",
        entry_point="bd1_env:BD1Env",
        max_episode_steps=500,
        autoreset=True,
    )  # TODO play with max_episode_steps

    args = parser.parse_args()

    env = gym.make("BD1_env", render_mode=None)
    # Create directories to hold models and logs
    model_dir = args.name
    log_dir = "logs/" + args.name
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train(
        env,
        args.algo,
        pretrained=args.pretrained,
        model_dir=model_dir,
        log_dir=log_dir,
        device=args.device,
    )

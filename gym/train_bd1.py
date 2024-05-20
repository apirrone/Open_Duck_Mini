import argparse
import os

import gymnasium as gym
from gymnasium.envs.registration import register
from sb3_contrib import TQC
from stable_baselines3 import A2C, SAC, TD3

register(
    id="BD1_env", entry_point="bd1_env:BD1Env", max_episode_steps=500, autoreset=True
)  # TODO play with max_episode_steps

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


def train(env, sb3_algo, pretrained=None):
    if pretrained is None:
        match sb3_algo:
            case "SAC":
                model = SAC(
                    "MlpPolicy", env, verbose=1, device="cuda", tensorboard_log=log_dir
                )
            case "TD3":
                model = TD3(
                    "MlpPolicy", env, verbose=1, device="cuda", tensorboard_log=log_dir
                )
            case "A2C":
                model = A2C(
                    "MlpPolicy", env, verbose=1, device="cuda", tensorboard_log=log_dir
                )
            case "TQC":
                model = TQC(
                    "MlpPolicy", env, verbose=1, device="cuda", tensorboard_log=log_dir
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
    args = parser.parse_args()

    env = gym.make("BD1_env", render_mode=None)

    train(env, args.algo, pretrained=args.pretrained)

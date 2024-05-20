import argparse
import os
from glob import glob

import gymnasium as gym
from gymnasium.envs.registration import register
from sb3_contrib import TQC
from stable_baselines3 import A2C, SAC, TD3

register(
    id="BD1_env", entry_point="bd1_env:BD1Env", autoreset=True, max_episode_steps=200
)


def test(env, sb3_algo, path_to_model):

    if not path_to_model.endswith(".zip"):
        models_paths = glob(path_to_model + "/*.zip")
        latest_model_id = 0
        latest_model_path = None
        for model_path in models_paths:
            model_id = model_path.split("/")[-1][: -len(".zip")].split("_")[-1]
            if int(model_id) > latest_model_id:
                latest_model_id = int(model_id)
                latest_model_path = model_path

        if latest_model_path is None:
            print("No models found in directory: ", path_to_model)
            return

        print("Using latest model: ", latest_model_path)

        path_to_model = latest_model_path

    match sb3_algo:
        case "SAC":
            model = SAC.load(path_to_model, env=env)
        case "TD3":
            model = TD3.load(path_to_model, env=env)
        case "A2C":
            model = A2C.load(path_to_model, env=env)
        case "TQC":
            model = TQC.load(path_to_model, env=env)

        case _:
            print("Algorithm not found")
            return

    obs = env.reset()[0]
    done = False
    extra_steps = 500
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)

        if done:
            extra_steps -= 1

            if extra_steps < 0:
                break


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test model.")
    parser.add_argument(
        "-p",
        "--path",
        metavar="path_to_model",
        help="Path to the model. If directory, will use the latest model.",
    )
    parser.add_argument("-a", "--algo", default="SAC")
    args = parser.parse_args()

    gymenv = gym.make("BD1_env", render_mode="human")
    test(gymenv, args.algo, path_to_model=args.path)

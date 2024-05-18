import argparse
import os

import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import A2C, SAC, TD3

register(
    id="BD1_env",
    entry_point="bd1_env:BD1Env",
)


def test(env, sb3_algo, path_to_model):

    match sb3_algo:
        case "SAC":
            model = SAC.load(path_to_model, env=env)
        case "TD3":
            model = TD3.load(path_to_model, env=env)
        case "A2C":
            model = A2C.load(path_to_model, env=env)
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

    parser = argparse.ArgumentParser(description="Train or test model.")
    parser.add_argument("-p", "--path", metavar="path_to_model")
    args = parser.parse_args()

    gymenv = gym.make("BD1_env", render_mode="human")
    test(gymenv, "SAC", path_to_model=args.path)

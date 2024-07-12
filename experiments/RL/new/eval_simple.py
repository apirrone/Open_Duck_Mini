import argparse
from scipy.spatial.transform import Rotation as R
import os
from glob import glob

import cv2
import FramesViewer.utils as fv_utils
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium.envs.registration import register
from sb3_contrib import TQC
from stable_baselines3 import A2C, PPO, SAC, TD3

register(
    id="BDX_env",
    entry_point="simple_env:BDXEnv",
    autoreset=True,
    # max_episode_steps=200,
)


def draw_clock(clock):
    # clock [a, b]
    clock_radius = 100
    im = np.zeros((clock_radius * 2, clock_radius * 2, 3), np.uint8)
    im = cv2.circle(im, (clock_radius, clock_radius), clock_radius, (255, 255, 255), -1)
    im = cv2.line(
        im,
        (clock_radius, clock_radius),
        (
            int(clock_radius + clock_radius * clock[0]),
            int(clock_radius + clock_radius * clock[1]),
        ),
        (0, 0, 255),
        2,
    )
    cv2.imshow("clock", im)
    cv2.waitKey(1)


def draw_frame(pose, i, env):
    pose = fv_utils.rotateInSelf(pose, [0, 90, 0])
    # env.mujoco_renderer._get_viewer(render_mode="human")
    env.mujoco_renderer._get_viewer(render_mode="human").add_marker(
        pos=pose[:3, 3],
        mat=pose[:3, :3],
        size=[0.005, 0.005, 0.1],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        rgba=[1, 0, 0, 1],
        label=str(i),
    )


def draw_velocities(robot_orig_xy, velocities_xytheta, env):
    horizon = 10  # seconds

    robot_orig_xyz = np.array([robot_orig_xy[0], robot_orig_xy[1], 0])
    for i in range(horizon):
        j = i * 0.1
        frame = np.eye(4)
        frame[:3, 3] = robot_orig_xyz + np.array(
            [velocities_xytheta[0] * j, velocities_xytheta[1] * j, 0]
        )
        # rotate frame to point in the direction of the velocity

        frame = fv_utils.rotateAbout(
            frame,
            [0, 0, velocities_xytheta[2] * j],
            center=robot_orig_xyz,
            degrees=False,
        )

        draw_frame(frame, i, env)


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
        case "PPO":
            # model = PPO.load(path_to_model, env=env)

            model = PPO("MlpPolicy", env)
            model.policy.load(path_to_model)
        case _:
            print("Algorithm not found")
            return

    obs = env.reset()[0]
    done = False
    extra_steps = 500
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)

        draw_velocities(env.data.body("base").xpos[:2], env.target_velocities, env)

        if done:
            extra_steps -= 1

            if extra_steps < 0:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model")
    parser.add_argument(
        "-p",
        "--path",
        metavar="path_to_model",
        help="Path to the model. If directory, will use the latest model.",
    )
    parser.add_argument("-a", "--algo", default="SAC")
    args = parser.parse_args()

    gymenv = gym.make("BDX_env", render_mode="human")
    test(gymenv, args.algo, path_to_model=args.path)

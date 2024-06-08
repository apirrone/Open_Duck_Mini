import argparse
import time
from glob import glob

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium.envs.registration import register
from stable_baselines3 import PPO, SAC

from mini_bdx.utils.mujoco_utils import check_contact


def get_observation(data, left_contact, right_contact):

    position = (
        data.qpos.flat.copy()
    )  # position/rotation of trunk + position of all joints
    velocity = (
        data.qvel.flat.copy()
    )  # positional/angular velocity of trunk +  of all joints

    obs = np.concatenate(
        [
            position,
            velocity,
            [left_contact, right_contact],
        ]
    )
    # print("OBS SIZE", len(obs))
    return obs


def key_callback(keycode):
    pass


def get_model_from_dir(path_to_model):

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
    else:
        latest_model_path = path_to_model

    return latest_model_path


def get_feet_contact(data, model):
    right_contact = check_contact(data, model, "foot_module", "floor")
    left_contact = check_contact(data, model, "foot_module_2", "floor")
    return right_contact, left_contact


def play(env, path_to_model):
    model_path = get_model_from_dir(path_to_model)

    model = mujoco.MjModel.from_xml_path("../../mini_bdx/robots/bdx/scene.xml")
    data = mujoco.MjData(model)

    left_contact = False
    right_contact = False

    viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)

    # nn_model = SAC.load(model_path, env)

    nn_model = PPO("MlpPolicy", env)
    nn_model.policy.load(model_path)

    try:
        while True:

            right_contact, left_contact = get_feet_contact(data, model)
            obs = get_observation(
                data,
                left_contact,
                right_contact,
            )
            action, _ = nn_model.predict(obs)
            data.ctrl[:] = action

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep / 2.5)

    except KeyboardInterrupt:
        viewer.close()

    viewer.close()


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

    register(id="BDX_env", entry_point="env_humanoid:BDXEnv")
    env = gym.make("BDX_env", render_mode=None)
    play(env, path_to_model=args.path)

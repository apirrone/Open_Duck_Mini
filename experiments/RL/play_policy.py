import argparse
import time
from glob import glob

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium.envs.registration import register
from stable_baselines3 import PPO


def get_observation(
    data,
    joint_history_length,
    joint_error_history,
    joint_ctrl_history,
    target_velocity,
    left_contact,
    right_contact,
):

    joints_rotations = data.qpos[7 : 7 + 13]
    joints_velocities = data.qvel[6 : 6 + 13]

    joints_error = data.ctrl - data.qpos[7 : 7 + 13]
    joint_error_history.append(joints_error)
    joint_error_history = joint_error_history[-joint_history_length:]

    angular_velocity = data.body("base").cvel[
        :3
    ]  # TODO this is imu, add noise to it later
    linear_velocity = data.body("base").cvel[3:]

    joint_ctrl_history.append(data.ctrl.copy())
    joint_ctrl_history = joint_ctrl_history[-joint_history_length:]

    return np.concatenate(
        [
            joints_rotations,
            joints_velocities,
            angular_velocity,
            linear_velocity,
            target_velocity,
            np.array(joint_error_history).flatten(),
            [left_contact, right_contact],
            [data.time],
        ]
    )


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


def play(env, path_to_model):
    model_path = get_model_from_dir(path_to_model)

    model = mujoco.MjModel.from_xml_path("../../mini_bdx/robots/bdx/scene.xml")
    data = mujoco.MjData(model)

    joint_history_length = 3
    joint_error_history = joint_history_length * [13 * [0]]
    joint_ctrl_history = joint_history_length * [13 * [0]]
    target_velocity = np.zeros(3)
    left_contact = False
    right_contact = False

    viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)

    nn_model = PPO("MlpPolicy", env)
    nn_model.policy.load(model_path)

    try:
        while True:
            obs = get_observation(
                data,
                joint_history_length,
                joint_error_history,
                joint_ctrl_history,
                target_velocity,
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

    register(id="BDX_env", entry_point="env:BDXEnv")
    env = gym.make("BDX_env", render_mode=None)
    play(env, path_to_model=args.path)

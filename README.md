# mini_BD1

## Install 
    
```bash
$ pip install -r requirements.txt
```


## Get get robot urdf from onshape
```bash
$ onshape-to-robot robots/bd1/
```

## Convert urdf to Mujoco

Download mujoco binaries somewhere https://github.com/google-deepmind/mujoco/releases

unpack and run

```bash
$ ./compile input.urdf output.xml
```

### Add actuators : 
Example : 
```xml
	<actuator>
		<!-- User parameter is the maximum no-load motor RPM -->
		<motor name="left_hip_yaw" joint="left_hip_yaw" gear="25" ctrlrange="-0.698132 0.698132" user="2900" />
		<motor name="left_hip_roll" joint="left_hip_roll" gear="25" ctrlrange="-1.5708 0.349066" user="2900" />
		<motor name="left_hip_pitch" joint="left_hip_pitch" gear="16" ctrlrange="1.0472 3.14159" user="1300" />
		<motor name="left_knee" joint="left_knee" gear="16" ctrlrange="-2.0944 2.0944" user="1300" />
		<motor name="left_ankle" joint="left_ankle" gear="50" ctrlrange="-1.5708 1.5708" user="5500" />
		<motor name="right_hip_roll" joint="right_hip_roll" gear="25" ctrlrange="-1.5708 0.349066" user="2900" />
		<motor name="right_hip_yaw" joint="right_hip_yaw" gear="25" ctrlrange="-0.698132 0.698132" user="2900" />
		<motor name="right_hip_pitch" joint="right_hip_pitch" gear="16" ctrlrange="-0.523599 1.5708" user="1300" />
		<motor name="right_knee" joint="right_knee" gear="16" ctrlrange="-2.0944 2.0944" user="1300" />
		<motor name="right_ankle" joint="right_ankle" gear="50" ctrlrange="-1.5708 1.5708" user="5500" />
	</actuator>
```

### Add a freejoint 

encapsulate the body in a freejoint
```xml
<worldbody>
	<body>
		<freejoint />
		...
		...
	</body>
</worldbody>
```

### Setup collision groups
Put that inside the <mujoco> bracket
```xml
<mujoco>
	<default>
		<geom contype="1" conaffinity="1" />
	</default>
	...
	...
</mujoco>
```

still need to add : 
- change frames to sites

## Visualize 

```bash
$ python3 -m mujoco.viewer --mjcf=<path>/scene.xml
```

Or 

```bash
$ cd mujoco
$ python visu.py -p <path_to_xml>
```

## RL stuff

there is a known bug with mujoco when trying to render the environment,

https://github.com/Farama-Foundation/Gymnasium/issues/749

To fix

in `<venv_path>gymnasium/envs/mujoco/mujoco_rendering.py` line 592 change solver_iter to solver_niter

### Train 

```bash
$ cd gym
$ python train_bd1.py -a <algo> -p <[optional]path_to_pretrained_model>
```

#### To Monitor during training

```bash
$ tensorboard --logdir=logs
```

### Infer

```bash
$ cd gym
$ python test_bd1.py -a <algo> -p <path_to_model>
```

## TODO
- Try a simpler task first, like juste balancing
  - Or try with a static goal
  - Or try with distance walked as reward
- Work on reward
  - Stop the episode when the robot falls
- Look at examples envs
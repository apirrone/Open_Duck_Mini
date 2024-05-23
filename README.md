# mini_BD1

## Install 
    
```bash
$ pip install -e .
```

## Make sure to design you robot according to onshape-to-robot constraints
https://onshape-to-robot.readthedocs.io/en/latest/design.html

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

### Setup collision groups, damping and friction
remove actuatorfrcrange in joints
Put that inside the <mujoco> bracket
```xml
<mujoco>
  <default>
    <geom contype="1" conaffinity="1" solref=".004 1" />
    <joint damping="0.09" frictionloss="0.1"/>
    <position kp="21.1" forcerange="-5.0 5.0"/>
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
- try flat feet with additional ankle roll joint ?
- make foot look more like a BDX (rubber contact)
  - Can I specify a contact material in mujoco ?

## Long term TODO
- Make it look cool like a BD-X or a BD-1
- nice close up video https://www.youtube.com/watch?v=QuWaaNN-1hs

## TO read
- https://www.nature.com/articles/s41598-023-38259-7.pdf
- https://arxiv.org/pdf/2304.13653
- https://arxiv.org/pdf/2401.16889
- https://arxiv.org/pdf/1801.01290
- First comment of this video explains the reward https://www.youtube.com/watch?v=L_4BPjLBF4E

# Ideas

# References 
- https://cults3d.com/en/3d-model/gadget/robot-disney-star-wars
- https://www.disneytouristblog.com/wp-content/uploads/2023/10/bd1-droid-testing-disney-imagineering-2.jpg

# Motors 
Prendre les M288T
https://en.robotis.com/shop_en/item.php?it_id=902-0163-000
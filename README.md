# Mini BDX Droid

![Capture d’écran du 2024-05-23 20-31-53](https://github.com/apirrone/mini_BDX/assets/6552564/d9b59a42-511d-40e9-96fc-b4fe29384bfe)


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
		<position name="left_hip_yaw"    joint="left_hip_yaw"    inheritrange="1"/>
		<position name="left_hip_roll"   joint="left_hip_roll"   inheritrange="1"/>
		<position name="left_hip_pitch"  joint="left_hip_pitch"  inheritrange="1"/>
		<position name="left_knee"       joint="left_knee"       inheritrange="1"/>
		<position name="left_ankle"      joint="left_ankle"      inheritrange="1"/>
		<position name="right_hip_roll"  joint="right_hip_roll"  inheritrange="1"/>
		<position name="right_hip_yaw"   joint="right_hip_yaw"   inheritrange="1"/>
		<position name="right_hip_pitch" joint="right_hip_pitch" inheritrange="1"/>
		<position name="right_knee"      joint="right_knee"      inheritrange="1"/>
		<position name="right_ankle"     joint="right_ankle"     inheritrange="1"/>
		<position name="head_pitch1"     joint="head_pitch1"     inheritrange="1"/>
		<position name="head_pitch2"     joint="head_pitch2"     inheritrange="1"/>
		<position name="head_yaw"        joint="head_yaw"        inheritrange="1"/>
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
/!\ remove actuatorfrcrange in joints
Put that inside the <mujoco> bracket
```xml
<mujoco>
  <default>
    <geom contype="1" conaffinity="1" solref=".004 1" />
    <joint damping="0.09" frictionloss="0.1"/>
    <position kp="10" forcerange="-5.0 5.0"/>
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
$ python train.py -a <algo> -n <experiment_name> -p <[optional]path_to_pretrained_model>
```

#### To Monitor during training

```bash
$ tensorboard --logdir=logs
```

### Infer

```bash
$ cd gym
$ python test.py -a <algo> -p <path_to_model>
```

## TODO
- try flat feet with additional ankle roll joint ?

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

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
still need to add : 
- change frames to sites

## Visualize 

```bash
$ cd mujoco
$ python visu.py -p <path_to_xml>
```
## TODO  

- [] fix zeros in onshape
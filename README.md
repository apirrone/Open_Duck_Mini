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
$ compile input.urdf output.xml
```

still need to add : 
- actuators
- change frames to sites
- add a floating base

## TODO  

- [] fix zeros in onshape
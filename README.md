# Mini BDX Droid

![Capture d’écran du 2024-05-23 20-31-53](https://github.com/apirrone/mini_BDX/assets/6552564/d9b59a42-511d-40e9-96fc-b4fe29384bfe)


## Install 
    
```bash
$ pip install -e .
```


## Visualize 

```bash
$ python3 -m mujoco.viewer --mjcf=<path>/scene.xml
```

or 

```bash

$ <path_to_mujoco_bin>/bin/simulate <path>/scene.xml
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

### TODO Placo walk engine
- tune the hell out of it
  - Get walk wit no step size working
- Understand it properly
  - Work out the transition between not moving, stepping without step size and stepping with step size
- Ramp up step size
- Pid on trunk pitch and head x position to stabilize 
  - Already added trunk_pitch = -imu_pitch*2

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

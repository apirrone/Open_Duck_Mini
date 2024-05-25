# Mini BDX Droid

I'm making a miniature version of the BDX Droid by Disney. It will be about 35 centimeters tall with its legs extended.

https://github.com/apirrone/mini_BDX/assets/6552564/1b8fb545-8831-40ba-97de-3689f4c851d3

## CAD 

https://cad.onshape.com/documents/a18ff8cc622a533762a3a6f5/w/27ef6089ad5fe9ba396b6036/e/9ce9b71d7a21eb04415b067f

## Install 
    
See [this document](docs/prepare_robot.md) for getting from a onshape design to a simulated robot in MuJoCo

```bash
$ pip install -e .
```

## RL stuff

there is a known bug with mujoco when trying to render the environment,

https://github.com/Farama-Foundation/Gymnasium/issues/749

To fix

in `<venv_path>gymnasium/envs/mujoco/mujoco_rendering.py` line 592 change solver_iter to solver_niter

### Train 

```bash
$ python train.py -a <algo> -n <experiment_name> -p <[optional]path_to_pretrained_model>
```

#### To Monitor during training

```bash
$ tensorboard --logdir=logs
```

### Infer

```bash
$ python test.py -a <algo> -p <path_to_model>
```

# TODO

## Design TODO
- Rework the roll/pitch part of the hips, its very ugly right now
  - Reworked a little, still ugly
- Integrate contact sensors in the feet
- Make the head part to link the pitch and yaw
- Make it look cool like a BD-X or a BD-1

# TO read
- https://www.nature.com/articles/s41598-023-38259-7.pdf
- https://arxiv.org/pdf/2304.13653
- https://arxiv.org/pdf/2401.16889
- https://arxiv.org/pdf/1801.01290
- First comment of this video explains the reward https://www.youtube.com/watch?v=L_4BPjLBF4E

# Ideas
- Try to make the action space of the rl algo to be the parameters of the expert walk ? 
  - frequency
  - trunk_pitch
  - trunk_roll
  - trunk_yaw
  - trunk_x_offset
  - step_size_x
  - step_size_y
  - rise_gain
  - swing_gain
- OR make the action space of the rl algo to be the cartesian pose of the feet in the trunk frame?

# References 
- https://cults3d.com/en/3d-model/gadget/robot-disney-star-wars
- https://www.disneytouristblog.com/wp-content/uploads/2023/10/bd1-droid-testing-disney-imagineering-2.jpg
- nice close up video https://www.youtube.com/watch?v=QuWaaNN-1hs

# Motors 
- M288T https://en.robotis.com/shop_en/item.php?it_id=902-0163-000

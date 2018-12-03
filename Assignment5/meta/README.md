# CS294-112 HW 5c: Meta-Learning

Dependencies:
 * Python **3.5**
 * Numpy version 1.14.5
 * TensorFlow version 1.10.5
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**
 * seaborn
 * Box2D==2.3.2

See the [HW5c PDF](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw5c.pdf) for further instructions.

# Instructions for Assignment5
## Problem 1: Context as Task ID

### 1. Training 

```ruby
python train_policy.py ’pm-obs’ --exp_name p1 --history 1 -lr 5e-5 -n 200 --num_tasks 4
```

### 2. Plot Result

```ruby
python plot.py logdir data/p1_*
```


## Problem 2: Meta-Learned Context
### 1. Training

```ruby
python train_policy.py ’pm’ --exp_name p2_60_r --history 60 --discount 0.90 -
```

### 2. Plot Result

```ruby
python plot.py logdir data/p2_*
```

## Problem 3: Generalization

### 1. Training

```ruby
python train_policy.py ’pm’ --exp_name p3_g1 --history 60 --discount 0.90 -lr 5e-4 -n 60 -rec -tt --gran 1
```
### 2. Plot Result

```ruby
python plot.py logdir data/p3_g1_* --value AverageReturn ValAverageReturn
```



## Cheers!
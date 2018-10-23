# CS294-112 HW 4: Model-based RL

Dependencies:
 * Python **3.5**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**
 * seaborn
 * Box2D==**2.3.2**
 * OpenCV
 * ffmpeg

Obtain the code from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw4. In addition to the installation requirements from previous homeworks, install additional required packages by running:
```rubypip install -r requirements.txt
```

See the [HW4 PDF](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw4.pdf) for further instructions.

The starter code was provided by OpenAI.

# Instructions for Assignment4
## Part 1: Dynamics Model

### 1. Training 

```ruby
python main.py q1
```


## Part 2: Action Selection
### 1. Training

```ruby
python main.py q2
```

## Part 3: On-Policy Data Collection

### Question a: Basic Algorithm Performance
#### 1. Training

```ruby
python main.py q3 --exp_name default
```
#### 2. Plot Result

```ruby
python plot.py --exps HalfCheetah_q3_default --save HalfCheetah_q3_default
```

### Question b: Experimenting Parameters

- Number of Random Action Sequences

#### 1. Training

```ruby
python main.py q3 --exp_name action128 --num_random_action_selection 128
python main.py q3 --exp_name action4096 --num_random_action_selection 4096
python main.py q3 --exp_name action16384 --num_random_action_selection 16384
```
#### 2. Plot Result

```ruby
python plot.py --exps HalfCheetah_q3_action128 HalfCheetah_q3_action4096 HalfCheetah_q3_action16384 --save HalfCheetah_q3_actions
```

- MPC planning horizon

#### 1. Training

```ruby
python main.py q3 --exp_name horizon10 --mpc_horizon 10
python main.py q3 --exp_name horizon15 --mpc_horizon 15
python main.py q3 --exp_name horizon20 --mpc_horizon 20
```
#### 2. Plot Result

```ruby
python plot.py --exps HalfCheetah_q3_horizon10 HalfCheetah_q3_horizon15 HalfCheetah_q3_horizon20 --save HalfCheetah_q3_mpc_horizon
```

- Number of neural network layers

#### 1. Training

```ruby
python main.py q3 --exp_name layers1 --nn_layers 1
python main.py q3 --exp_name layers2 --nn_layers 2
python main.py q3 --exp_name layers3 --nn_layers 3
```
#### 2. Plot Result

```ruby
python plot.py --exps HalfCheetah_q3_layers1 HalfCheetah_q3_layers2 HalfCheetah_q3_layers3 --save HalfCheetah_q3_nn_layers
```



## Cheers!
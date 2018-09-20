# CS294-112 HW 1: Imitation Learning

Dependencies:
 * Python **3.5**
 * Numpy
 * TensorFlow
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**

Once Python **3.5** is installed, you can install the remaining dependencies using `pip install -r requirements.txt`.

**Note**: MuJoCo versions until 1.5 do not support NVMe disks therefore won't be compatible with recent Mac machines.
There is a request for OpenAI to support it that can be followed [here](https://github.com/openai/gym/issues/638).

**Note**: Students enrolled in the course will receive an email with their MuJoCo activation key. Please do **not** share this key.

The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* Ant-v2.pkl
* HalfCheetah-v2.pkl
* Hopper-v2.pkl
* Humanoid-v2.pkl
* Reacher-v2.pkl
* Walker2d-v2.pkl

The name of the pickle file corresponds to the name of the gym environment.


# Instructions for Assignment1

Environment:
OSX 10.11.6 + Anaconda3 + Python 3.6 + Tensorflow 1.8.0 + Gym 0.10.5

### 1. Use pretrained model to see the result

- Section2 Question2

```ruby
python sec2q2.py Reacher-v2 --num_rollouts1=400 Humanoid-v2 --num_rollouts2=20 --nonlinear=True
```

- Section2 Question3

```ruby
python sec2q3.py Humanoid-v2 --num_rollouts=20 --nonlinear=True
```

- Section3 Question2

```ruby
python sec3q2.py Humanoid-v2 --num_rollouts=20 --nonlinear=True
```

- Section4 Question1

```ruby
python sec4q1.py Reacher-v2 --num_rollouts=400 --nonlinear=False
```

### 2. Rerun code to get model and test

- Section2 Question2

##### 1) For Reacher-v2:

Run expert to generate data

```ruby
python run_expert.py experts/Reacher-v2.pkl Reacher-v2 --render --num_rollouts=400
```

Run behavioral cloning

```ruby
python behavioral_cloning.py Reacher-v2 --num_rollouts=400 --nonlinear=True --restore=False --iter_seed=False
```

Run evaluation of behavioral cloning

```ruby
python evaluation.py Reacher-v2 --render --num_rollouts=400 --nonlinear=True --iter_seed=False
```

##### 2) For Humanoid-v2:

Run expert to generate data

```ruby
python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --render --num_rollouts=20
```

Run behavioral cloning

```ruby
python behavioral_cloning.py Humanoid-v2 --num_rollouts=20 --nonlinear=True --restore=False --iter_seed=False
```

Run evaluation of behavioral cloning

```ruby
python evaluation.py Humanoid-v2 --render --num_rollouts=20 --nonlinear=True --iter_seed=False
```

##### 3) Get result

```ruby
python sec2q2.py Reacher-v2 --num_rollouts1=400 Humanoid-v2 --num_rollouts2=20 --nonlinear=True
```

- Section2 Question3

Iterate different seeds for behavioral cloning

```ruby
python behavioral_cloning.py Humanoid-v2 --num_rollouts=20 --nonlinear=True --restore=False --iter_seed=True
```

Iterate different seeds for evaluation of behavioral cloning

```ruby
python evaluation.py Humanoid-v2 --num_rollouts=20 --nonlinear=True --iter_seed=True
```

Show result

```ruby
python sec2q3.py Humanoid-v2 --num_rollouts=20 --nonlinear=True
```

- Section3 Question2

Run DAgger to generate reward data

```ruby
python DAgger.py Humanoid-v2 experts/Humanoid-v2.pkl --num_rollouts 20 --nonlinear True
```

Show result

```ruby
python sec3q2.py Humanoid-v2 --num_rollouts=20 --nonlinear=True
```

- Section4 Question1

Iterate different seeds for Reacher-v2 task with nonlinear activation layer

```ruby
python behavioral_cloning.py Reacher-v2 --num_rollouts=400 --nonlinear=True --restore=False --iter_seed=True
python evaluation.py Reacher-v2 --num_rollouts=400 --nonlinear=True --iter_seed=True
```

Iterate different seeds for Reacher-v2 task without nonlinear activation layer

```ruby
python behavioral_cloning.py Reacher-v2 --num_rollouts=400 --nonlinear=False --restore=False --iter_seed=True
python evaluation.py Reacher-v2 --num_rollouts=400 --nonlinear=False --iter_seed=True
```

Show result

```ruby
python sec4q1.py Reacher-v2 --num_rollouts=400 --nonlinear=False
```

## Cheers!


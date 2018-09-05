Instructions for assignment 1_Yujie Wang_2018.09.03

Environment:
OSX 10.11.6 + Anaconda3 + Python 3.6 + Tensorflow 1.8.0 + Gym 0.10.5

1. Use pretrained model to see the result

## Section2 Question2
python sec2q2.py Reacher-v2 --num_rollouts1=400 Humanoid-v2 --num_rollouts2=20 --nonlinear=True

## Section2 Question3
python sec2q3.py Humanoid-v2 --num_rollouts=20 --nonlinear=True

## Section3 Question2
python sec3q2.py Humanoid-v2 --num_rollouts=20 --nonlinear=True

## Section4 Question1
python sec4q1.py Reacher-v2 --num_rollouts=400 --nonlinear=False



2. Rerun code to get model and test

## Section2 Question2

1) For Reacher-v2:

# Run expert to generate data
python run_expert.py experts/Reacher-v2.pkl Reacher-v2 --render --num_rollouts=400

# Run behavioral cloning
python behavioral_cloning.py Reacher-v2 --num_rollouts=400 --nonlinear=True --restore=False --iter_seed=False

# Run evaluation of behavioral cloning
python evaluation.py Reacher-v2 --render --num_rollouts=400 --nonlinear=True --iter_seed=False

2) For Humanoid-v2:

# Run expert to generate data
python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --render --num_rollouts=20

# Run behavioral cloning
python behavioral_cloning.py Humanoid-v2 --num_rollouts=20 --nonlinear=True --restore=False --iter_seed=False

# Run evaluation of behavioral cloning
python evaluation.py Humanoid-v2 --render --num_rollouts=20 --nonlinear=True --iter_seed=False

3) Get result

python sec2q2.py Reacher-v2 --num_rollouts1=400 Humanoid-v2 --num_rollouts2=20 --nonlinear=True


## Section2 Question3

# Iterate different seeds for behavioral cloning
python behavioral_cloning.py Humanoid-v2 --num_rollouts=20 --nonlinear=True --restore=False --iter_seed=True

# Iterate different seeds for evaluation of behavioral cloning
python evaluation.py Humanoid-v2 --num_rollouts=20 --nonlinear=True --iter_seed=True

# Show result
python sec2q3.py Humanoid-v2 --num_rollouts=20 --nonlinear=True


## Section3 Question2

# Run DAgger to generate reward data
python DAgger.py Humanoid-v2 experts/Humanoid-v2.pkl --num_rollouts 20 --nonlinear True

# Show result
python sec3q2.py Humanoid-v2 --num_rollouts=20 --nonlinear=True


## Section4 Question1

# Iterate different seeds for Reacher-v2 task with nonlinear activation layer
python behavioral_cloning.py Reacher-v2 --num_rollouts=400 --nonlinear=True --restore=False --iter_seed=True
python evaluation.py Reacher-v2 --num_rollouts=400 --nonlinear=True --iter_seed=True

# Iterate different seeds for Reacher-v2 task without nonlinear activation layer
python behavioral_cloning.py Reacher-v2 --num_rollouts=400 --nonlinear=False --restore=False --iter_seed=True
python evaluation.py Reacher-v2 --num_rollouts=400 --nonlinear=False --iter_seed=True

# Show result
python sec4q1.py Reacher-v2 --num_rollouts=400 --nonlinear=False


Please contact me (yujie_wang@berkeley.edu) if you encounter any problem running code. Thanks for reading.

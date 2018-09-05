# Experiment how behavioral cloning performance different from expert in two tasks
import numpy as np
import matplotlib . pyplot as plt 
import seaborn as sns
import os
import pickle
import tensorflow as tf
from viz_utilize import reward_result

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('envname1', type=str)
parser.add_argument('--num_rollouts1', type=int, default=20,
                    help = 'Number of expert roll outs 1')
parser.add_argument('envname2', type=str)
parser.add_argument('--num_rollouts2', type=int, default=20,
                    help = 'Number of expert roll outs 2')
parser.add_argument('--nonlinear', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
args = parser.parse_args()
# Parameters
seed_expert = 0
seed_BC = 2
dagger_iter = 6
seed_DA_start = 4
seed_DA_end = 7

## Task Reacher
print("Task Reacher ................")

# Load Expert log data
reacher_expert_path = os.path.join('expert_data',args.envname1 +'-'+str(args.num_rollouts1)+'-seed'+str(seed_expert)+'.pkl')
reacher_expert_reward = reward_result(reacher_expert_path)
print('expert reward mean ',np.mean(reacher_expert_reward))
print('expert reward std ',np.std(reacher_expert_reward))

# Load Behavioral  Cloning log data
reacher_BC_path = os.path.join('log/', args.envname1 +'-' + str(args.num_rollouts1) + '-seed'+ str(seed_BC)+ '-reward.pkl')
reacher_BC_reward = reward_result(reacher_BC_path)
print('Behavioral  Cloning reward mean ',np.mean(reacher_BC_reward))
print('Behavioral  Cloning reward std ',np.std(reacher_BC_reward))

## Task Humanoid
print("Task Humanoid ................")

# Load Expert log data
humanoid_expert_path = os.path.join('expert_data',args.envname2 +'-'+str(args.num_rollouts2)+'-seed'+str(seed_expert)+'.pkl')
humanoid_expert_reward = reward_result(humanoid_expert_path)
print('expert reward mean ',np.mean(humanoid_expert_reward))
print('expert reward std ',np.std(humanoid_expert_reward))

# Load Behavioral  Cloning log data
humanoid_BC_path = os.path.join('log/', args.envname2 +'-' + str(args.num_rollouts2) + '-seed'+ str(seed_BC)+ '-reward.pkl')
humanoid_BC_reward = reward_result(humanoid_BC_path)
print('Behavioral  Cloning reward mean ',np.mean(humanoid_BC_reward))
print('Behavioral  Cloning reward std ',np.std(humanoid_BC_reward))
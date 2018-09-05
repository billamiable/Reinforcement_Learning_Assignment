# Experiment how activation layer affect BC performance
import numpy as np
import matplotlib . pyplot as plt 
import seaborn as sns
import os
import pickle
import tensorflow as tf
from viz_utilize import reward_result

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('envname', type=str)
parser.add_argument('--num_rollouts', type=int, default=20,
                    help = 'Number of expert roll outs 1')
parser.add_argument('--nonlinear', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
args = parser.parse_args()
print(args.nonlinear)

# Parameter
seed_BC_start = 2
seed_BC_end = 5

# Load nonlinear log data
nonlinear_rewards = []
for seed_BC in range(seed_BC_start, seed_BC_end):
    BC_path = os.path.join('log/', args.envname +'-' + str(args.num_rollouts) + '-seed'+ str(seed_BC)+ '-reward.pkl')
    BC_reward = reward_result(BC_path)
    nonlinear_rewards.append(BC_reward)
print(np.shape(nonlinear_rewards))

# Load linear log data
linear_rewards = []
for seed_BC in range(seed_BC_start, seed_BC_end):
    BC_path_linear = os.path.join('log/', args.envname +'-' + str(args.num_rollouts) + '-seed'+ str(seed_BC)+ '-linear-reward.pkl')
    BC_reward_linear = reward_result(BC_path_linear)
    linear_rewards.append(BC_reward_linear)
print(np.shape(linear_rewards))

plt.figure()
# Plot error bars for different seed
seed_BC = [i for i in range(seed_BC_start, seed_BC_end)] 
# Plot nonlinear method
plt.errorbar(seed_BC, np.mean(nonlinear_rewards,axis=1),  yerr=np.std(nonlinear_rewards,axis=1), fmt='o', markersize=8, capsize=20)
# Plot linear method
plt.errorbar(seed_BC, np.mean(linear_rewards,axis=1),  yerr=np.std(linear_rewards,axis=1), fmt='o', markersize=8, capsize=20)
# Text and Caption
plt.ylabel("Rewards", fontsize=15)
plt.xlabel("Seed Number", fontsize=15, labelpad=4)
plt.xlim((1,5))
plt.legend(labels=["wi activation layer","wo activation layer"], loc='lower right')
plt.title ("Behavioral Cloning Performance wi/wo activation layer for Reacher", fontsize=20)
plt.subplots_adjust(bottom=0.27) # Leave space between xlabel and figtext
txt = """Task: Reacher-v2, Results Reproducible
    Network with or without activation layer (Consider variability: seed number = 2, 3, 4)
    Want to see for simple task like Reacher, simple linear regression can also work.
    But the result turns out that activation layer enhances performance in this setting.
    Training -- Training epochs: 500, Learning rate: 0.001, Batch size: 64
    Data setting -- Split: 80% Train, 20% Test, Normalization: True, Shuffle: True
    Network Structure -- Hidden Layer #: 2, Hidden neurons #: 64/layer, Activation: ReLu or None
    Data Amount = 20000 = 400(Reacher_num_rollouts)*50(Reacher_sample#/rollout)
    """
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=10)
plt.show()
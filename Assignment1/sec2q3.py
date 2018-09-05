# Experiment with hyperparameter seed
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

# Make result reproducible
seed_expert = 0
seed_BC_start = 2
seed_BC_end = 5

# Load log data
rewards = []
for seed_BC in range(seed_BC_start, seed_BC_end):
    BC_path = os.path.join('log/', args.envname +'-' + str(args.num_rollouts) + '-seed'+ str(seed_BC)+ '-reward.pkl')
    BC_reward = reward_result(BC_path)
    rewards.append(BC_reward)

# Load Expert log data for comparison
humanoid_expert_path = os.path.join('expert_data',args.envname +'-'+str(args.num_rollouts)+'-seed'+str(seed_expert)+'.pkl')
humanoid_expert_reward = reward_result(humanoid_expert_path)
humanoid_expert_reward = np.tile(humanoid_expert_reward,(seed_BC_end-seed_BC_start,1))

plt.figure()
# Plot error bars for different seed
seed_BC = [i for i in range(seed_BC_start, seed_BC_end)] 
plt.errorbar(seed_BC, np.mean(rewards,axis=1),  yerr=np.std(rewards,axis=1), fmt='o', markersize=8, capsize=20)
# Plot Expert
sns.tsplot(time=seed_BC, data=np.transpose(humanoid_expert_reward), color='k', linestyle='-.')
# Text and Caption
plt.ylabel("Rewards", fontsize=15)
plt.xlabel("Seed Number", fontsize=15, labelpad=4)
plt.xlim((1,5))
plt.legend(labels=["expert","BC"], loc='lower right')
plt.title ("behavioral Cloning Performance with different seed for Humanoid", fontsize=20)
plt.subplots_adjust(bottom=0.27) # Leave space between xlabel and figtext
txt = """Task: Humanoid-v2, Results Reproducible
    Seed variation: seed number = 2, 3, 4 (Comparison: straight line - expert)
    Seed determines network initialization, shuffle, batch sequence, initial gym observation,
    which can generate very different results, even make Humanoid BC's reward close to expert.
    Training -- Training epochs: 500, Learning rate: 0.001, Batch size: 64
    Data setting -- Split: 80% Train, 20% Test, Normalization: True, Shuffle: True
    Network Structure -- Hidden Layer #: 2, Hidden neurons #: 64/layer, Activation: ReLu
    Data Amount = 20000 = 20(Humanoid_num_rollouts)*1000(Humanoid_sample#/rollout)
    """
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=10)
plt.show()
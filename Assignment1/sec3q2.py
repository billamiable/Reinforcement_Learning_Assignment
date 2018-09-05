# Experiment DAgger performance for Humanoid
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

# Parameters
seed_expert = 0
seed_BC = 2
dagger_iter = 6
seed_DA_start = 4
seed_DA_end = 7

print("Task Humanoid ................")

# Load Expert log data
humanoid_expert_path = os.path.join('expert_data',args.envname +'-'+str(args.num_rollouts)+'-seed'+str(seed_expert)+'.pkl')
humanoid_expert_reward = reward_result(humanoid_expert_path)
# Duplicate to meet viz form
humanoid_expert_reward = np.tile(humanoid_expert_reward,(dagger_iter,1))
if 0:
    print(humanoid_expert_reward)
    print(np.shape(humanoid_expert_reward))
    exit()

# Load Behavioral Cloning log data
humanoid_BC_path = os.path.join('log/', args.envname +'-' + str(args.num_rollouts) + '-seed'+ str(seed_BC)+ '-reward.pkl')
humanoid_BC_reward = reward_result(humanoid_BC_path)
# Duplicate to meet viz form
humanoid_BC_reward = np.tile(humanoid_BC_reward,(dagger_iter,1))
if 0:
    print(humanoid_BC_reward)
    print(np.shape(humanoid_BC_reward))
    exit()

# Load DAgger log data
humanoid_DA_reward = []
for seed_DA in range(seed_DA_start, seed_DA_end):
    DA_reward = []
    for iter in range(0,dagger_iter):
        with open (os.path.join('log-DAgger/', args.envname +'-' + str(args.num_rollouts) + '-seed'+ str(seed_DA)+'-iter'+str(iter)+ '-reward.pkl'), 'rb') as fo:
            DA_data = pickle.load(fo)
            DA_reward.append(DA_data['rewards'])
    humanoid_DA_reward.append(DA_reward)
if 0:
    # (20,6)
    print(np.shape(np.transpose(DA_rewards[0])))
    exit()


# Plot error bars
plt.figure()
iter = [i for i in range(1,np.shape(humanoid_DA_reward)[1]+1)] 
# Plot Expert
sns.tsplot(time=iter, data=np.transpose(humanoid_expert_reward), color='k', linestyle='-.')

# Plot Behavioral Cloning
sns.tsplot(time=iter, data=np.transpose(humanoid_BC_reward), color='orange', linestyle='-.')

# Plot DAgger 
sns.tsplot(time=iter, data=np.transpose(humanoid_DA_reward[0]), color='r', linestyle='-')
sns.tsplot(time=iter, data=np.transpose(humanoid_DA_reward[1]), color='g', linestyle='--')
sns.tsplot(time=iter, data=np.transpose(humanoid_DA_reward[2]), color='b', linestyle=':') 

# Text and Caption
plt.ylabel("Rewards", fontsize=15)
plt.xlabel("Iteration Number", fontsize=15, labelpad=4)
plt.title ("DAgger Performance for Humanoid", fontsize=20)
plt.legend(labels=["expert","BC","DA_seed=4", "DA_seed=5", "DA_seed=6"], loc='lower right')
plt.subplots_adjust(bottom=0.23) # Leave space between xlabel and figtext
txt = """Task: Humanoid-v2, Results Reproducible
    DAgger Setting -- DAgger_iter: 6, DAgger_num_rollouts/iter: 13
    Training -- Training epochs: 50/DAgger_iter, Learning rate: 0.001, Batch size: 64
    Data setting -- Split: 80% Train, 20% Test, Normalization: True, Shuffle: True
    Network Structure -- Hidden Layer #: 2, Hidden neurons #: 64/layer, Activation: ReLu
    Data Amount≈20000(previous)+6(DAgger_iter)*13(DAgger_num_rollouts/iter)*1000(sample#/rollout)=98000
    """
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=10)
plt.show()

if 0:
    # For test
    with open (os.path.join('log/', args.envname +'-' + str(args.num_rollouts) + '.pkl'), 'rb') as fo:
        loss_data = pickle.load(fo)

    # This is just a dummy function to generate some arbitrary data 
    def get_data():
        base_cond = [[18 ,20 ,19 ,18 ,13 ,4 ,1] , [20,17,12,9,3,0,0], [20 ,20 ,20 ,12 ,5 ,3 ,0]]
        cond1 = [[18 ,19 ,18 ,19 ,20 ,15 ,14] , [19 ,20 ,18 ,16 ,20 ,15 ,9] , [19 ,20 ,20 ,20 ,17 ,10 ,0] , [20 ,20 ,20 ,20 ,7 ,9 ,1]]
        cond2 = [[20 ,20 ,20 ,20 ,19 ,17 ,4] , [20 ,20 ,20 ,20 ,20 ,19 ,7] , [19 ,20 ,20 ,19 ,19 ,15 ,2]]
        cond3 = [[20 ,20 ,20 ,20 ,19 ,17 ,12] , [18,20,19,18,13,4,1], [20,19,18,17,13,2,0], [19 ,18 ,20 ,20 ,15 ,6 ,0]]
        return base_cond , cond1 , cond2 , cond3

    # Load the data . 
    results = get_data() 
    fig = plt.figure()
    # We will plot iterations 0 ... 6 
    xdata = np. array ([0 ,1 ,2 ,3 ,4 ,5 ,6])/5.
    # Plot each line
    # (may want to automate this part e.g. with a loop).
    sns.tsplot(time=xdata, data=results[0], color='r', linestyle='-')
    sns.tsplot(time=xdata, data=results[1], color='g', linestyle='--') 
    sns.tsplot(time=xdata, data=results[2], color='b', linestyle=':') 
    sns.tsplot(time=xdata, data=results[3], color='k', linestyle='-.')


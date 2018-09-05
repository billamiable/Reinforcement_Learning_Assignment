import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from network import neural_net
from simulate import simulate

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('envname', type=str)
parser.add_argument('--render', action='store_true')
parser.add_argument("--max_timesteps", type=int)
parser.add_argument('--num_rollouts', type=int, default=20,
                    help = 'Number of expert roll outs')
parser.add_argument('--nonlinear', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument('--iter_seed', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
args = parser.parse_args()
print(args.nonlinear)
print(args.iter_seed)

# Make result reproducible
seed_expert = 0
seed_BC_start = 2
if args.iter_seed:
    seed_BC_end = 5
else:
    seed_BC_end = 3

for seed_BC in range(seed_BC_start, seed_BC_end):
    print('current seed is ', seed_BC)
    np.random.seed(seed_BC)
    tf.set_random_seed(seed_BC)

    # Prepare dataset
    with open(os.path.join('expert_data', args.envname + '-' + str(args.num_rollouts)+ '-seed' + str(seed_expert) + '.pkl'), 'rb') as f:
        data = pickle.load(f) # dict
    mean = np.mean(data['observations'], axis=0)
    std = np.std(data['observations'], axis=0) + 1e-6

    # Different path for linear/nonlinear settings
    if args.nonlinear:
        # Restore model graph
        model_path = os.path.join('model/', args.envname +'-' + str(args.num_rollouts)+ '-seed'+ str(seed_BC)+ '/model.ckpt')
        meta_path = os.path.join('model/', args.envname +'-' + str(args.num_rollouts)+ '-seed'+ str(seed_BC)+ '/model.ckpt.meta')
        # Save reward path
        save_path = os.path.join('log/', args.envname +'-' + str(args.num_rollouts) + '-seed'+ str(seed_BC)+ '-reward.pkl')
    else:
        model_path = os.path.join('model/', args.envname +'-' + str(args.num_rollouts)+ '-seed'+ str(seed_BC)+ '-linear/model.ckpt')
        meta_path = os.path.join('model/', args.envname +'-' + str(args.num_rollouts)+ '-seed'+ str(seed_BC)+ '-linear/model.ckpt.meta')
        save_path = os.path.join('log/', args.envname +'-' + str(args.num_rollouts) + '-seed'+ str(seed_BC)+ '-linear-reward.pkl')

    # Evaluation
    policy_path, is_dagger, save_oa = False, False, False
    observations, actions, rewards = simulate(args.envname, args.render, args.max_timesteps, args.num_rollouts, policy_path, model_path, meta_path, save_path, mean, std, seed_BC, is_dagger, save_oa)
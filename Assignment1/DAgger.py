import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from network import neural_net
from training import training
from simulate import simulate

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('envname', type=str)
parser.add_argument('expert_policy_file', type=str)
parser.add_argument('--render', action='store_true')
parser.add_argument("--max_timesteps", type=int)
parser.add_argument('--num_rollouts', type=int, default=20,
                    help = 'Number of expert roll outs')
parser.add_argument('--nonlinear', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
args = parser.parse_args()

# Parameters
dagger_iter = 6
learning_rate = 0.001 
training_epochs = 50
batch_size = 64
display_step = 10
if args.nonlinear:
    n_hidden_1 = 64 # 1st layer number of neurons
    n_hidden_2 = 64 # 2nd layer number of neurons

# Make results reproducible
seed_expert = 0
seed_BC = 2
seed_DA_start = 4
seed_DA_end = 7

for seed_DA in range(seed_DA_start, seed_DA_end):
    # Make result reproducible
    print('current seed is ', seed_DA)
    print('*************************************')
    np.random.seed(seed_DA)
    tf.set_random_seed(seed_DA)
    
    # Prepare dataset
    with open(os.path.join('expert_data', args.envname + '-' + str(args.num_rollouts)+ '-seed' + str(seed_expert) + '.pkl'), 'rb') as f:
        data = pickle.load(f) # dict
    input_dim = np.shape(data['observations'])[1]
    output_dim = np.shape(data['actions'])[1]
    print('input dim is', input_dim)
    print('output dim is', output_dim)
    new_obs = data['observations']
    new_act = data['actions']

    # Restore model graph
    saver = tf.train.import_meta_graph(os.path.join('model/', args.envname +'-' + str(args.num_rollouts)+ '-seed'+ str(seed_BC)+ '/model.ckpt.meta'))
    cost = tf.get_collection("training_collection")[0]
    pred = tf.get_collection("training_collection")[1]
    optimizer = tf.get_collection("training_collection")[2]

    # Start DAgger loop
    for iter in range(dagger_iter):
        print('iteration', iter, '........................')

        # Reset Tensorflow Graph
        tf.reset_default_graph()

        # Construct model
        cost, pred, optimizer = neural_net(args.nonlinear, input_dim, output_dim, n_hidden_1, n_hidden_2, seed_DA, batch_size, learning_rate)
        print('Constructed nerual network.')
        # Add useful variables to collection
        tf.add_to_collection("training_collection", cost)
        tf.add_to_collection("training_collection", pred)
        tf.add_to_collection("training_collection", optimizer)

        # Train model
        dagger_path = os.path.join('model-DAgger/', args.envname +'-' + str(args.num_rollouts) + '-seed'+ str(seed_DA)+ '/model.ckpt')
        if iter == 0: # First load model without DAgger
            restore_path = os.path.join('model/', args.envname +'-' + str(args.num_rollouts)+ '-seed'+ str(seed_BC)+ '/model.ckpt')
        else: # Then load previous DAgger model
            restore_path = dagger_path
        model_save_path = dagger_path
        loss_save_path = os.path.join('log-DAgger/', args.envname +'-' + str(args.num_rollouts) + '-seed'+ str(seed_DA)+'-iter'+str(iter)+'-loss.pkl')
        loss, mean, std = training(new_obs, new_act, restore_path, cost, pred, optimizer, 0.8, True, True, training_epochs, batch_size, display_step, model_save_path, loss_save_path)

        # Run gym and obtain expert action for new observation
        meta_path = os.path.join('model-DAgger/', args.envname +'-' + str(args.num_rollouts) + '-seed'+ str(seed_DA)+ '/model.ckpt.meta')
        model_path = dagger_path
        save_path = os.path.join('log-DAgger/', args.envname +'-' + str(args.num_rollouts) + '-seed'+ str(seed_DA)+'-iter'+str(iter)+ '-reward.pkl')
        num_rollouts = int(args.num_rollouts*4 / dagger_iter)
        is_dagger, save_oa = True, False
        observations, actions, rewards, actions_expert = simulate(args.envname, args.render, args.max_timesteps, num_rollouts, args.expert_policy_file, model_path, meta_path, save_path, mean, std, seed_DA, is_dagger, save_oa)

        # Merge dataset
        new_obs =  np.concatenate((new_obs, np.array(observations)), axis=0)
        new_act =  np.concatenate((new_act, np.array(actions_expert)), axis=0) 
        # print(np.shape(observations))
        # print(np.shape(new_obs))
# implement Behavioral cloning_2018.8.26_Yujie
# Reference1: https://github.com/aymericdamien/TensorFlow-Examples/
# Reference2: TF_lecture.ipynb

import os
import pickle
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from network import neural_net
from training import training

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('envname', type=str)
parser.add_argument('--num_rollouts', type=int, default=20,
                    help = 'Number of expert roll outs')
parser.add_argument('--nonlinear', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument('--restore', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument('--iter_seed', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
args = parser.parse_args()
print(args.nonlinear)
print(args.restore)
print(args.iter_seed)

# Parameters
learning_rate = 0.001 
training_epochs = 500
batch_size = 64
display_step = 50
n_hidden_1 = 64 # 1st layer number of neurons
n_hidden_2 = 64 # 2nd layer number of neurons

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
    with open(os.path.join('expert_data', args.envname +'-' + str(args.num_rollouts) + '-seed' + str(seed_expert) + '.pkl'), 'rb') as f:
        data = pickle.load(f)
    input_dim = np.shape(data['observations'])[1]
    output_dim = np.shape(data['actions'])[1]
    print('input dim is', input_dim)
    print('output dim is', output_dim)

    # Reset Tensorflow Graph
    tf.reset_default_graph()

    # Construct model # Restore it is useless to construct??
    cost, pred, optimizer = neural_net(args.nonlinear, input_dim, output_dim, n_hidden_1, n_hidden_2, seed_BC, batch_size, learning_rate)
    print('Constructed nerual network.')
    # Add useful variables to collection
    tf.add_to_collection("training_collection", cost)
    tf.add_to_collection("training_collection", pred)
    tf.add_to_collection("training_collection", optimizer)
    if 0:
        # Restore variable from meta graph
        saver = tf.train.import_meta_graph(os.path.join('model/', args.envname +'-' + str(args.num_rollouts)+ '/model.ckpt.meta'))
        cost = tf.get_collection("training_collection")[0]
        pred = tf.get_collection("training_collection")[1]
        optimizer = tf.get_collection("training_collection")[2]
    
    # Different path for linear/nonlinear settings
    if args.nonlinear:
        path = os.path.join('model/', args.envname +'-' + str(args.num_rollouts) + '-seed'+ str(seed_BC)+ '/model.ckpt')
        loss_save_path = os.path.join('log/', args.envname +'-' + str(args.num_rollouts) + '-seed'+ str(seed_BC)+ '-loss.pkl')
    else:
        path = os.path.join('model/', args.envname +'-' + str(args.num_rollouts) + '-seed'+ str(seed_BC)+ '-linear/model.ckpt')
        loss_save_path = os.path.join('log/', args.envname +'-' + str(args.num_rollouts) + '-seed'+ str(seed_BC)+ '-linear-loss.pkl')
    
    # Restore model or not
    if args.restore:
        restore_path = path
    else:
        restore_path = False

    # Train model
    model_save_path = path
    loss, mean, std = training(data['observations'], data['actions'], restore_path, cost, pred, optimizer, 0.8, True, True, training_epochs, batch_size, display_step, model_save_path, loss_save_path)

if 0:
    # Visualization
    epoch = [i for i in range(1,np.shape(loss['train'])[0]+1)] 
    plt.plot(epoch, loss['train'])
    plt.plot(epoch, loss['test'])
    plt.legend(['Train error', 'Test error'], loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Imitation Learning Before DAgger')
    plt.show()
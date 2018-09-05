import tensorflow as tf
import numpy as np

# Network design
def neural_net(nonlinear, input_dim, output_dim, n_hidden_1, n_hidden_2, seed, batch_size, learning_rate):   
    x = tf.placeholder(tf.float32, [None, input_dim], name='x')
    y = tf.placeholder(tf.float32, [None, output_dim], name='y')

    with tf.variable_scope('layer_0'):
        W0 = tf.get_variable(name='W0', shape=[input_dim, n_hidden_1], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        b0 = tf.get_variable(name='b0', shape=[n_hidden_1], initializer=tf.constant_initializer(0.))

    with tf.variable_scope('layer_1'):
        W1 = tf.get_variable(name='W1', shape=[n_hidden_1, n_hidden_2], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        b1 = tf.get_variable(name='b1', shape=[n_hidden_2], initializer=tf.constant_initializer(0.))
        
    with tf.variable_scope('layer_2'):
        W2 = tf.get_variable(name='W2', shape=[n_hidden_2, output_dim], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        b2 = tf.get_variable(name='b2', shape=[output_dim], initializer=tf.constant_initializer(0.))
    
    # # print the variables
    # var_names = sorted([v.name for v in tf.global_variables()])
    # print('\n'.join(var_names))

    weights = [W0, W1, W2]
    biases = [b0, b1, b2]
    if nonlinear:
        activations = [tf.nn.relu, tf.nn.relu, None]
    else:
        activations = [None, None, None]

    # Create model
    layer = x
    for W, b, activation in zip(weights, biases, activations):
        layer = tf.matmul(layer, W) + b
        if activation is not None:
            layer = activation(layer)
    pred = layer


    # Calculate cost
    cost = tf.reduce_sum(tf.pow(pred-y, 2))/(0.5 * batch_size) # Mean square error
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

    return cost, pred, optimizer
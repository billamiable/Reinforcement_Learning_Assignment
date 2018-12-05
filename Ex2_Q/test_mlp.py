import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers

def ex2_model(img_in, scope, reuse=False, dropout=False, keep_prob=1.0):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            #out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
            #if dropout:
            #    out = layers.dropout(out, keep_prob=keep_prob)
            out = layers.fully_connected(out, num_outputs=1, activation_fn=None)

        return out
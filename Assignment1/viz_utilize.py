import numpy as np
import matplotlib . pyplot as plt 
import seaborn as sns
import os
import pickle
import tensorflow as tf

def reward_result(path):
    with open (path, 'rb') as fo:
        data = pickle.load(fo)
        rewards = data['rewards']
    return rewards



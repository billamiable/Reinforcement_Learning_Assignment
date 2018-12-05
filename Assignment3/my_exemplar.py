from fan import model
import numpy as np
import tensorflow as tf


class Exemplar(object):
    def __init__(
            self,
            train_itrs=1e3,
            first_train_itrs=5e3,
            min_replay_size=1e3,    
        ):

    def fit(self, postives, negatives):
        if self.replay.size >= self.min_replay_size:
            #log_step = self.train_itrs * self.log_freq
            labels = np.expand_dims(np.concatenate([np.ones(self.batch_size), np.zeros(self.batch_size)]), 1).astype(np.float32)

            if self.first_train:
                train_itrs = self.first_train_itrs
                self.first_train = False
            else:
                train_itrs = self.train_itrs

            for train_itr in range(train_itrs):
                pos_batch = sample_batch(positives, positives.shape[0], self.batch_size)
                neg_batch = self.replay.random_batch(self.batch_size)
                x1 = np.concatenate([pos_batch, pos_batch])
                x2 = np.concatenate([pos_batch, neg_batch])
                loss, class_loss, kl_loss = self.model.train_batch(x1, x2, labels)

    def predict(self, path, negatives):
        if self.replay.size < self.min_replay_size:
            return np.zeros(len(path['observations']))
        counts = self.model.test(positives)


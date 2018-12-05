from fan import MODEL
import numpy as np
import tensorflow as tf


class Exemplar(object):
    def __init__(
            self,
            bonus_form="1/sqrt(p)",
        ):
        self.first_train = False
        self.bonus_form = bonus_form
        self.model = MODEL()

    def fit(self, postives, negatives):
        #log_step = self.train_itrs * self.log_freq
        labels = np.expand_dims(np.concatenate([np.ones(self.batch_size), np.zeros(self.batch_size)]), 1).astype(np.float32)
        pos_batch = sample_batch(positives, positives.shape[0], self.batch_size)
        neg_batch = self.replay.random_batch(self.batch_size)
        x1 = np.concatenate([pos_batch, pos_batch])
        x2 = np.concatenate([pos_batch, neg_batch])
        loss, class_loss, kl_loss = self.model.train_batch(x1, x2, labels)

    def predict(self, path, negatives):
        counts = self.model.test(positives)
        # if self.rank == 0:
        #     logger.record_tabular('Average Prob', np.mean(counts))
        #     logger.record_tabular('Average Discrim', np.mean(1/(5.01*counts + 1)))

        if self.bonus_form == "1/n":
            bonuses = 1./counts
        elif self.bonus_form == "1/sqrt(pn)":
            bonuses = 1. / np.sqrt(self.replay.size * counts)
        elif self.bonus_form == "1/sqrt(p)":
            bonuses = 1./np.sqrt(counts)
        elif self.bonus_form == "1/log(n+1)":
            bonuses = 1./np.log(counts + 1)
        elif self.bonus_form == "1/log(n)":
            bonuses = 1. / np.log(counts)
        elif self.bonus_form == "-log(p)":
            bonuses = - np.log(counts)
        else:
            raise NotImplementedError
        return bonuses

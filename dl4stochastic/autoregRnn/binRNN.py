# Autoregressive RNN with logistic outputs.
# Author: Yingru Liu
# Institute: Stony Brook University

import tensorflow as tf
from .utility import hidden_net, config

class binRNN():

    def __init__(
            self,
            Config,
    ):
        if Config.dimLayer == []:
            raise(ValueError('The structure is empty!'))
        if Config.dimLayer[-1] != Config.dimLayer[0]:
            Config.dimLayer[-1] = Config.dimLayer[0]

        self.x = tf.placeholder(dtype='float32', shape=[Config.batch_size, Config.max_steps, Config.dimLayer[0]])
        self._numLayer = len(Config.dimLayer) - 2

        # Build the Inference Network
        self._cell, hiddenOutput, initializer = hidden_net(self.x, Config)
        with tf.variable_scope('logit', initializer=initializer):
            W = tf.get_variable('weight', shape=(Config.dimLayer[-2], Config.dimLayer[-1]))
            b = tf.get_variable('bias', shape=Config.dimLayer[-1], initializer=tf.zeros_initializer)
            logits = tf.nn.xw_plus_b(hiddenOutput, W, b)
            logits = tf.reshape(logits, [Config.batch_size, Config.max_steps, Config.dimLayer[-1]])
            self._outputs = tf.nn.sigmoid(logits)
        # define the loss function.
            self._loss = tf.losses.sigmoid_cross_entropy(self.x[:, 1:, :], logits[:, 0:-1, :])
            self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        self._sess = tf.Session
        self._runSession()

    def _runSession(self):
        self._sess.run(tf.global_variables_initializer())

    def train_function(self):
        pass

    def val_function(self):
        pass

    def gen_function(self):
        pass

    def saver(self):
        pass
# Autoregressive RNN with logistic outputs.
# Author: Yingru Liu
# Institute: Stony Brook University

import tensorflow as tf
from .utility import hidden_net, config
import numpy as np

# General Auto-regressive RNN
class arRNN(object):
    def __init__(
            self,
            Config,
    ):
        if Config.dimLayer == []:
            raise(ValueError('The structure is empty!'))
        if Config.dimLayer[-1] != Config.dimLayer[0]:
            Config.dimLayer[-1] = Config.dimLayer[0]

        self.x = tf.placeholder(dtype='float32', shape=[Config.batch_size, Config.max_steps, Config.dimLayer[0]])
        self.lr = tf.placeholder(dtype='float32', shape=(), name='learningRate')
        self._numLayer = len(Config.dimLayer) - 2
        self._dimLayer = Config.dimLayer
        self._savePath = Config.savePath
        self._eventPath = Config.eventPath

        # Build the Inference Network
        self._cell, self._hiddenOutput, self._initializer = hidden_net(self.x, Config)
        if Config.Opt == 'Adadelta':
            self._optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
        elif Config.Opt == 'Adam':
            self._optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        elif Config.Opt == 'Momentum':
            self._optimizer = tf.train.MomentumOptimizer(self.lr, 0.5)
        elif Config.Opt == 'SGD':
            self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        else:
            self._optimizer = None
            raise(ValueError("Config.Opt should be either 'Adadelta', 'Adam', 'Momentum' or 'SGD'!"))

        self._sess = tf.Session()
        self._saver = tf.train.Saver()

    def _runSession(self):
        self._sess.run(tf.global_variables_initializer())

    def train_function(self, input, lrate):
        return

    def val_function(self):
        return

    def gen_function(self):
        return

    def saver(self, summaryEnable=False, feed_dict=None, summaryClose=False):
        self._saver.restore(self._sess, self._savePath)
        if summaryEnable and feed_dict is not None:
            summary = tf.summary.merge_all()
            summary_str = self._sess.run(summary, feed_dict=feed_dict)
            self._summary.add_summary(summary_str)
            self._summary.flush()
        if summaryClose:
            self._summary.close()

    def full_train(self):
        pass

class binRNN(arRNN):

    def __init__(
            self,
            Config,
    ):
        arRNN.__init__(self, Config)

        with tf.variable_scope('logit', initializer=self._initializer):
            # define the output layer.
            W = tf.get_variable('weight', shape=(Config.dimLayer[-2], Config.dimLayer[-1]))
            b = tf.get_variable('bias', shape=Config.dimLayer[-1], initializer=tf.zeros_initializer)
            logits = tf.nn.xw_plus_b(self._hiddenOutput, W, b)
            logits = tf.reshape(logits, [Config.batch_size, Config.max_steps, Config.dimLayer[-1]])
            self._outputs = tf.nn.sigmoid(logits)
            # define the loss function.
            self._loss = tf.losses.sigmoid_cross_entropy(self.x[:, 1:, :], logits[:, 0:-1, :])
            self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self._train_step = self._optimizer.minimize(self._loss)
            self._runSession()
            self._summary = tf.summary.FileWriter(self._eventPath, self._sess.graph)

    def train_function(self, input, lrate):
        zero_padd = np.zeros(shape=(input.shape[0], 1, input.shape[2]), dtype='float32')
        con_input = np.concatenate((zero_padd, input), axis=1)
        _, loss_value = self._sess.run([self._train_step, self._loss],
                                       feed_dict={self.x: con_input, self.lr: lrate})
        return loss_value * input.shape[-1]

    def val_function(self, input):
        zero_padd = np.zeros(shape=(input.shape[0], 1, input.shape[2]), dtype='float32')
        con_input = np.concatenate((zero_padd, input), axis=1)
        loss_value = self._sess.run(self._loss, feed_dict={self.x: con_input})
        return loss_value * input.shape[-1]

    def gen_function(self, numSteps):
        state = self._cell.zero_state(1, dtype=tf.float32)
        x_ = tf.zeros((1, self._dimLayer[0]), dtype='float32')
        samples = []
        with tf.variable_scope('logit', reuse=True):
            W = tf.get_variable('weight')
            b = tf.get_variable('bias')
            for i in range(numSteps):
                hidde_, state = self._cell(x_, state)
                probs = tf.nn.sigmoid(tf.nn.xw_plus_b(hidde_, W, b))
                x_ = tf.distributions.Bernoulli(probs=probs, dtype=tf.float32).sample()
                samples.append(x_)
        samples = tf.concat(samples, 0)
        return self._sess.run(samples)

class gaussRNN(arRNN):
    pass

class gmmRNN(arRNN):
    pass
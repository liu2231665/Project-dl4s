"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: This file contains the Autoregressive RNN with arbitrary
              structures for both the binary and continuous inputs.
              ----2017.11.01
#########################################################################"""

import tensorflow as tf
from .utility import hidden_net
from dl4s.tools import get_batches_idx, GaussNLL
import numpy as np
import time

"""#########################################################################
Class: arRNN - the hyper abstraction of the auto-regressive RNN.
#########################################################################"""
class _arRNN(object):

    """#########################################################################
    __init__:the initialization function.
    input: Config - configuration class in ./utility.
    output: None.
    #########################################################################"""
    def __init__(
            self,
            Config,
    ):
        # Check the dimension configuration.
        if Config.dimLayer == []:
            raise(ValueError('The structure is empty!'))
        # Check the autoregressive structure(i.e. dim of output is equal to dim of input).
        if Config.dimLayer[-1] != Config.dimLayer[0]:
            Config.dimLayer[-1] = Config.dimLayer[0]

        # <tensor graph> define a default graph.
        self._graph = tf.Graph()

        with self._graph.as_default():
            # <tensor placeholder> input.
            self.x = tf.placeholder(dtype='float32', shape=[None, None, Config.dimLayer[0]])
            # <tensor placeholder> learning rate.
            self.lr = tf.placeholder(dtype='float32', shape=(), name='learningRate')

            # <scalar> number of hidden layers.
            self._numLayer = len(Config.dimLayer) - 2
            # <scalar list> dimensions of each layer[input, hiddens, output].
            self._dimLayer = Config.dimLayer
            # <string/None> path to save the model.
            self._savePath = Config.savePath
            # <string/None> path to save the events.
            self._eventPath = Config.eventPath
            # <string/None> path to load the events.
            self._loadPath = Config.loadPath
            # <list> collection of trainable parameters.
            self._params = []
            # <pass> will be define in the children classes.
            self._train_step = None
            # <pass> will be define in the children classes.
            self._loss = None
            # <pass> will be define in the children classes.
            self._outputs = None

            # Build the Inference Network
            # self._cell: the mutil - layer hidden cells.
            # self._hiddenOutput - the output with shape [batch_size, max_time, cell.output_size].
            # self._initializer - Initializer.
            self._cell, self._hiddenOutput, self._initializer = hidden_net(
                self.x, self._graph, Config)

            # <Tensorflow Optimizer>.
            if Config.Opt == 'Adadelta':
                self._optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
            elif Config.Opt == 'Adam':
                self._optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            elif Config.Opt == 'Momentum':
                self._optimizer = tf.train.MomentumOptimizer(self.lr, 0.9)
            elif Config.Opt == 'SGD':
                self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            else:
                raise(ValueError("Config.Opt should be either 'Adadelta', 'Adam', 'Momentum' or 'SGD'!"))
            # <Tensorflow Session>.
            self._sess = tf.Session(graph=self._graph)
        return

    """#########################################################################
    _runSession: initialize the graph or restore from the load path.
    input: None.
    output: None.
    #########################################################################"""
    def _runSession(self):
        if self._loadPath is None:
            self._sess.run(tf.global_variables_initializer())
        else:
            saver = tf.train.Saver()
            saver.restore(self._sess, self._loadPath)
        return

    """#########################################################################
    train_function: compute the loss and update the tensor variables.
    input: input - numerical input.
           lrate - <scalar> learning rate.
    output: the loss value.
    #########################################################################"""
    def train_function(self, input, lrate):
        with self._graph.as_default():
            zero_padd = np.zeros(shape=(input.shape[0], 1, input.shape[2]), dtype='float32')
            con_input = np.concatenate((zero_padd, input), axis=1)
            _, loss_value = self._sess.run([self._train_step, self._loss],
                                           feed_dict={self.x: con_input, self.lr: lrate})
        return loss_value * input.shape[-1]

    """#########################################################################
    val_function: compute the loss with given input.
    input: input - numerical input.
    output: the loss value.
    #########################################################################"""
    def val_function(self, input):
        with self._graph.as_default():
            zero_padd = np.zeros(shape=(input.shape[0], 1, input.shape[2]), dtype='float32')
            con_input = np.concatenate((zero_padd, input), axis=1)
            loss_value = self._sess.run(self._loss, feed_dict={self.x: con_input})
        return loss_value * input.shape[-1]

    """#########################################################################
    output_function: compute the output with given input.
    input: input - numerical input.
    output: the output values of the network.
    #########################################################################"""
    def output_function(self, input):
        with self._graph.as_default():
            zero_padd = np.zeros(shape=(input.shape[0], 1, input.shape[2]), dtype='float32')
            con_input = np.concatenate((zero_padd, input), axis=1)
            output = self._sess.run(self._outputs, feed_dict={self.x: con_input})
        return output[:, 0:-1, :]


"""#########################################################################
Class: binRNN - the auto-regressive Recurrent Neural Network for stochastic
                binary inputs.
#########################################################################"""
class binRNN(_arRNN, object):
    """
        __init__:the initialization function.
        input: Config - configuration class in ./utility.
        output: None.
    """
    def __init__(
            self,
            Config,
    ):
        _arRNN.__init__(self, Config)

        # add the output layer at the top of hidden output.
        with self._graph.as_default():
            with tf.variable_scope('logit', initializer=self._initializer):
                # define the output layer.
                W = tf.get_variable('weight', shape=(Config.dimLayer[-2], Config.dimLayer[-1]))
                b = tf.get_variable('bias', shape=Config.dimLayer[-1], initializer=tf.zeros_initializer)
                logits = tf.tensordot(self._hiddenOutput, W, [[-1], [0]]) + b
                self._outputs = tf.nn.sigmoid(logits)
                # define the loss function.
                self._loss = tf.losses.sigmoid_cross_entropy(self.x[:, 1:, :], logits[:, 0:-1, :])
                self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                self._train_step = self._optimizer.minimize(tf.cast(tf.shape(self.x), tf.float32)[-1]*self._loss)
                self._runSession()

    """#########################################################################
    gen_function: reconstruction of the gen_function in class: arRNN.
    #########################################################################"""
    def gen_function(self, numSteps):
        with self._graph.as_default():
            state = self._cell.zero_state(1, dtype=tf.float32)
            x_ = tf.zeros((1, self._dimLayer[0]), dtype='float32')
            samples = []
            with tf.variable_scope('logit', reuse=True):  # reuse the output layer.
                W = tf.get_variable('weight')
                b = tf.get_variable('bias')
                for i in range(numSteps):
                    hidde_, state = self._cell(x_, state)
                    probs = tf.nn.sigmoid(tf.nn.xw_plus_b(hidde_, W, b))
                    x_ = tf.distributions.Bernoulli(probs=probs, dtype=tf.float32).sample()
                    samples.append(x_)
            samples = tf.concat(samples, 0)
        return self._sess.run(samples)


"""#########################################################################
Class: gaussRNN - the auto-regressive Recurrent Neural Network for stochastic
                binary inputs.
#########################################################################"""
class gaussRNN(_arRNN, object):
    """
    __init__: the initialization function.
    input: Config - configuration class in ./ utility.
    output: None.
    """
    def __init__(
            self,
            Config,
    ):
        _arRNN.__init__(self, Config)

        # add the output layer at the top of hidden output.
        with self._graph.as_default():
            with tf.variable_scope('logit', initializer=self._initializer):
                # define the Gaussian output layer with diagonal layer.
                W_mu = tf.get_variable('weight_mu', shape=(Config.dimLayer[-2], Config.dimLayer[-1]))
                b_mu = tf.get_variable('bias_mu', shape=Config.dimLayer[-1], initializer=tf.zeros_initializer)
                W_sig = tf.get_variable('weight_sig', shape=(Config.dimLayer[-2], Config.dimLayer[-1]))
                b_sig = tf.get_variable('bias_sig', shape=Config.dimLayer[-1], initializer=tf.zeros_initializer)
                # mu - the mean of the conditional Gaussian distribution.
                # sig -  the variance of the conditional Gaussian distribution.
                #        (positive definiteness is assured by softplus function.)
                mu = tf.tensordot(self._hiddenOutput, W_mu, [[-1], [0]]) + b_mu
                sig = tf.nn.softplus(tf.tensordot(self._hiddenOutput, W_sig, [[-1], [0]]) + b_sig) + 1e-8
                self._outputs = [mu, sig]
                # define the loss function as negative log-likelihood.
                self._loss = GaussNLL(x=self.x[:, 1:, :], mean=mu[:, 0:-1, :], sigma=sig[:, 0:-1, :])
                self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                self._train_step = self._optimizer.minimize(tf.cast(tf.shape(self.x), tf.float32)[-1]*self._loss)
                self._runSession()

    """#########################################################################
    output_function: reconstruction of the output_function in class: arRNN.
    #########################################################################"""
    def output_function(self, input):
        with self._graph.as_default():
            zero_padd = np.zeros(shape=(input.shape[0], 1, input.shape[2]), dtype='float32')
            con_input = np.concatenate((zero_padd, input), axis=1)
            output = self._sess.run(self._outputs, feed_dict={self.x: con_input})
        return output[0][:, 0:-1, :], output[1][:, 0:-1, :]

    """#########################################################################
    gen_function: reconstruction of the gen_function in class: arRNN.
    #########################################################################"""
    def gen_function(self, numSteps):
        with self._graph.as_default():
            state = self._cell.zero_state(1, dtype=tf.float32)
            x_ = tf.zeros((1, self._dimLayer[0]), dtype='float32')
            samples = []
            with tf.variable_scope('logit', reuse=True):  # reuse the output layer.
                W_mu = tf.get_variable('weight_mu')
                b_mu = tf.get_variable('bias_mu')
                W_sig = tf.get_variable('weight_sig')
                b_sig = tf.get_variable('bias_sig')
                for i in range(numSteps):
                    hidde_, state = self._cell(x_, state)
                    mu = tf.nn.xw_plus_b(hidde_, W_mu, b_mu)
                    sig = tf.nn.softplus(tf.nn.xw_plus_b(hidde_, W_sig, b_sig)) + 1e-8
                    x_ = tf.distributions.Normal(loc=mu, scale=tf.sqrt(sig)).sample()
                    samples.append(x_)
            samples = tf.concat(samples, 0)
        return self._sess.run(samples)

"""
--------------------------------------------------------------------------------------------
"""
class gmmRNN(_arRNN):
    pass

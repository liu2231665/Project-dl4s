"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: This file contains the Autoregressive RNN with arbitrary
              structures for both the binary and continuous inputs.
              ----2017.11.01
#########################################################################"""

import tensorflow as tf
from .utility import hidden_net
from dl4s.cores.tools import GaussNLL
from dl4s.cores.model import _model
import numpy as np

"""#########################################################################
Class: arRNN - the hyper abstraction of the auto-regressive RNN.
#########################################################################"""
class _arRNN(_model, object):

    """#########################################################################
    __init__:the initialization function.
    input: Config - configuration class in ./utility.
    output: None.
    #########################################################################"""
    def __init__(
            self,
            config,
    ):

        # Check the dimension configuration.
        if config.dimLayer == []:
            raise(ValueError('The structure is empty!'))
        # Check the autoregressive structure(i.e. dim of output is equal to dim of input).
        config.dimLayer = [config.dimIN] + config.dimLayer + [config.dimIN]

        _model.__init__(self, config=config)
        with self._graph.as_default():
            # <scalar> number of hidden layers.
            self._numLayer = len(config.dimLayer) - 2
            # <scalar list> dimensions of each layer[input, hiddens, output].
            self._dimLayer = config.dimLayer

            # Build the Inference Network
            # self._cell: the mutil - layer hidden cells.
            # self._hiddenOutput - the output with shape [batch_size, max_time, cell.output_size].
            # self._initializer - Initializer.
            self._cell, self._hiddenOutput, self._initializer = hidden_net(
                self.x, self._graph, config)
            #
            self._feature = self._hiddenOutput
        return


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
                """define the loss and train_step."""
                self._loss = tf.losses.sigmoid_cross_entropy(self.x[:, 1:, :], logits[:, 0:-1, :])
                self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                self._train_step = self._optimizer.minimize(tf.cast(tf.shape(self.x), tf.float32)[-1]*self._loss)
                """define the process to generate samples."""
                # the initial state and initial input of the RNN.
                state = self._cell.zero_state(1, dtype=tf.float32)
                x_ = tf.zeros((1, self._dimLayer[0]), dtype='float32')
                # TensorArray to save the output of the generating.
                gen_operator = tf.TensorArray(tf.float32, self.sampleLen)
                # condition and body of while loop (input: i-iteration, xx-RNN input, ss-RNN state)
                i = tf.constant(0)
                cond = lambda i, xx, ss, array: tf.less(i, self.sampleLen)
                #
                def body(i, xx, ss, array):
                    ii = i + 1
                    hidde_, new_ss = self._cell(xx, ss)
                    probs = tf.nn.sigmoid(tf.nn.xw_plus_b(hidde_, W, b))
                    new_xx = tf.distributions.Bernoulli(probs=probs, dtype=tf.float32).sample()
                    new_array = array.write(i, new_xx)
                    return ii, new_xx, new_ss, new_array
                gen_operator = tf.while_loop(cond, body, [i, x_, state, gen_operator])[-1]
                self._gen_operator = gen_operator.concat()
                #
                self._runSession()


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
                # mu/sigma of the approximated probability
                self._prob = [mu, sig]
                self._outputs = mu
                """define the loss function as negative log-likelihood."""
                self._loss = GaussNLL(x=self.x[:, 1:, :], mean=mu[:, 0:-1, :], sigma=sig[:, 0:-1, :])
                self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                self._train_step = self._optimizer.minimize(tf.cast(tf.shape(self.x), tf.float32)[-1]*self._loss)
                """define the process to generate samples."""
                # the initial state and initial input of the RNN.
                state = self._cell.zero_state(1, dtype=tf.float32)
                x_ = tf.zeros((1, self._dimLayer[0]), dtype='float32')
                # TensorArray to save the output of the generating.
                gen_operator = tf.TensorArray(tf.float32, self.sampleLen)
                # condition and body of while loop (input: i-iteration, xx-RNN input, ss-RNN state)
                i = tf.constant(0)
                cond = lambda i, xx, ss, array: tf.less(i, self.sampleLen)
                #
                def body(i, xx, ss, array):
                    ii = i + 1
                    hidde_, new_ss = self._cell(xx, ss)
                    mu = tf.nn.xw_plus_b(hidde_, W_mu, b_mu)
                    sig = tf.nn.softplus(tf.nn.xw_plus_b(hidde_, W_sig, b_sig)) + 1e-8
                    new_xx = tf.distributions.Normal(loc=mu, scale=tf.sqrt(sig)).sample()
                    new_array = array.write(i, new_xx)
                    return ii, new_xx, new_ss, new_array

                gen_operator = tf.while_loop(cond, body, [i, x_, state, gen_operator])[-1]
                self._gen_operator = gen_operator.concat()
                #
                self._runSession()


"""
--------------------------------------------------------------------------------------------
"""
class gmmRNN(_arRNN):
    pass

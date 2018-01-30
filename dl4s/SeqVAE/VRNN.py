"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the file contains the model description of VRNN.
              ----2017.11.13
#########################################################################"""

import tensorflow as tf
from .utility import buildVRNN
from .utility import configVRNN
from . import GaussKL, BernoulliNLL, GaussNLL
from dl4s.tools import get_batches_idx
import numpy as np
import time, os

"""#########################################################################
Class: _VRNN - the hyper abstraction of the VRNN.
#########################################################################"""
class _VRNN(object):
    """#########################################################################
    __init__:the initialization function.
    input: Config - configuration class in ./utility.
    output: None.
    #########################################################################"""
    def __init__(
            self,
            config=configVRNN()
    ):
        # Check the dimension configuration.
        if config.dimRec == []:
            raise (ValueError('The recurrent structure is empty!'))
        # <tensor graph> define a default graph.
        self._graph = tf.Graph()
        with self._graph.as_default():
            # <tensor placeholder> input.
            self.x = tf.placeholder(dtype='float32', shape=[None, None, config.dimInput])
            # <tensor placeholder> learning rate.
            self.lr = tf.placeholder(dtype='float32', shape=(), name='learningRate')
            # <scalar list> the size of recurrent hidden layers.
            self._dimRec = config.dimRec
            # <scalar list> the size of feedforward hidden layers of input.
            self._dimForX = config.dimForX
            # <scalar list> the size of feedforward hidden layers of stochastic layer.
            self._dimForZ = config.dimForZ
            # <scalar> dimensions of input frame.
            self._dimInput = config.dimInput
            # <scalar> dimensions of stochastic states.
            self._dimState = config.dimState
            # <string/None> path to save the model.
            self._savePath = config.savePath
            # <string/None> path to save the events.
            self._eventPath = config.eventPath
            # <string/None> path to load the events.
            self._loadPath = config.loadPath
            # <list> collection of trainable parameters.
            self._params = []

            self._prior_mu, self._prior_sig, self._pos_mu, \
            self._pos_sig, self._hidden_dec, self._h_tm1, self._varCell,\
                self._Z = buildVRNN(self.x, self._graph, config)
            self._loss = GaussKL(self._pos_mu, self._pos_sig ** 2, self._prior_mu, self._prior_sig ** 2)
            self._kl_divergence = self._loss
            # <pass> will be define in the children classes.
            self._train_step = None
            # the prior P(Z).
            self._prior = [self._prior_mu, self._prior_sig]
            # the posterior P(Z|X).
            self._enc = [self._pos_mu, self._pos_sig]
            # <pass> compute the posterior P(X|Z)
            self._dec = []

            # <Tensorflow Optimizer>.
            if config.Opt == 'Adadelta':
                self._optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
            elif config.Opt == 'Adam':
                self._optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            elif config.Opt == 'Momentum':
                self._optimizer = tf.train.MomentumOptimizer(self.lr, 0.9)
            elif config.Opt == 'SGD':
                self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            else:
                raise (ValueError("Config.Opt should be either 'Adadelta', 'Adam', 'Momentum' or 'SGD'!"))
            # <Tensorflow Session>.
            self._sess = tf.Session(graph=self._graph)

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
            _, loss_value = self._sess.run([self._train_step, self._loss],
                                           feed_dict={self.x: input, self.lr: lrate})
        return loss_value * input.shape[-1]

    """#########################################################################
    val_function: compute the loss with given input.
    input: input - numerical input.
    output: the loss value.
    #########################################################################"""
    def val_function(self, input):
        with self._graph.as_default():
            loss_value = self._sess.run(self._loss, feed_dict={self.x: input})
        return loss_value * input.shape[-1]

    """#########################################################################
    encoder: return the mean and std of P(Z|X).
    input: input - numerical input.
    output: mean - mean.
            var - variance.
    #########################################################################"""
    def encoder(self, input):
        with self._graph.as_default():
            mean, var = self._sess.run(self._enc, feed_dict={self.x: input})
        return mean, var

"""#########################################################################
Class: binVRNN - the VRNN model for stochastic binary inputs.
#########################################################################"""
class binVRNN(_VRNN, object):
    """#########################################################################
    __init__:the initialization function.
    input: Config - configuration class in ./utility.
    output: None.
    #########################################################################"""
    def __init__(
            self,
            config=configVRNN
    ):
        super().__init__(config)
        with self._graph.as_default():
            with tf.variable_scope('logit'):
                Wdec = tf.get_variable('Wdec', shape=(self._hidden_dec.shape[-1], config.dimInput))
                bdec = tf.get_variable('bdec', shape=config.dimInput, initializer=tf.zeros_initializer)
                self._dec = tf.nn.sigmoid(tf.tensordot(self._hidden_dec, Wdec, [[-1], [0]]) + bdec)
                self._loss += BernoulliNLL(self.x, self._dec)
                self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                self._train_step = self._optimizer.minimize(tf.cast(tf.shape(self.x), tf.float32)[-1] * self._loss)
                self._runSession()

    """#########################################################################
    gen_function: generate samples.
    input: numSteps - the length of the sample sequence.
    output: should be the sample.
    #########################################################################"""
    def gen_function(self, numSteps):
        with self._graph.as_default():
            # Set the variational cell to use the prior P(Z) to generate Zt.
            self._varCell.setGen()
            #
            state = self._varCell.zero_state(1, dtype=tf.float32)
            x_ = tf.zeros((1, self._dimInput), dtype='float32')
            samples = []
            with tf.variable_scope('logit', reuse=True):
                Wdec = tf.get_variable('Wdec')
                bdec = tf.get_variable('bdec')

            for i in range(numSteps):
                (_, _, _, _, hidde_, _, _), state = self._varCell(x_, state)
                probs = tf.nn.sigmoid(tf.nn.xw_plus_b(hidde_, Wdec, bdec))
                x_ = tf.distributions.Bernoulli(probs=probs, dtype=tf.float32).sample()
                samples.append(x_)
            samples = tf.concat(samples, 0)
        return self._sess.run(samples)

    """#########################################################################
    output_function: reconstruction function.
    input: input - .
           samples - indicate wether outputs a sample.
    output: should be the reconstruction represented by the probability or the
            sample.
    #########################################################################"""
    def output_function(self, input, samples=True):
        with self._graph.as_default():
            if samples:
                return self._sess.run(tf.distributions.Bernoulli(probs=self._dec, dtype=tf.float32).sample(),
                                      feed_dict={self.x: input})
            else:
                return self._sess.run(self._dec, feed_dict={self.x: input})

"""#########################################################################
Class: gaussVRNN - the VRNN model for stochastic continuous inputs.
#########################################################################"""
class gaussVRNN(_VRNN, object):
    """#########################################################################
    __init__:the initialization function.
    input: Config - configuration class in ./utility.
    output: None.
    #########################################################################"""

    def __init__(
            self,
            config=configVRNN
    ):
        super().__init__(config)
        with self._graph.as_default():
            with tf.variable_scope('output'):
                # compute the mean and standard deviation of P(X|Z).
                Wdec_mu = tf.get_variable('Wdec_mu', shape=(self._hidden_dec.shape[-1], config.dimInput))
                bdec_mu = tf.get_variable('bdec_mu', shape=config.dimInput, initializer=tf.zeros_initializer)
                mu = tf.tensordot(self._hidden_dec, Wdec_mu, [[-1], [0]]) + bdec_mu
                Wdec_sig = tf.get_variable('Wdec_sig', shape=(self._hidden_dec.shape[-1], config.dimInput))
                bdec_sig = tf.get_variable('bdec_sig', shape=config.dimInput, initializer=tf.zeros_initializer)
                std = tf.nn.softplus(tf.tensordot(self._hidden_dec, Wdec_sig, [[-1], [0]]) + bdec_sig) + 1e-8
                self._dec = [mu, std]

                self._loss += GaussNLL(self.x, mu, std**2)
                self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                self._train_step = self._optimizer.minimize(tf.cast(tf.shape(self.x), tf.float32)[-1] * self._loss)
                self._runSession()

    """#########################################################################
    gen_function: generate samples.
    input: numSteps - the length of the sample sequence.
    output: should be the sample.
    #########################################################################"""
    def gen_function(self, numSteps):
        with self._graph.as_default():
            # Set the variational cell to use the prior P(Z) to generate Zt.
            self._varCell.setGen()
            #
            state = self._varCell.zero_state(1, dtype=tf.float32)
            x_ = tf.zeros((1, self._dimInput), dtype='float32')
            samples = []
            with tf.variable_scope('output', reuse=True):
                # compute the mean and standard deviation of P(X|Z).
                Wdec_mu = tf.get_variable('Wdec_mu')
                bdec_mu = tf.get_variable('bdec_mu')
                Wdec_sig = tf.get_variable('Wdec_sig')
                bdec_sig = tf.get_variable('bdec_sig')

            for i in range(numSteps):
                (_, _, _, _, hidde_, _, _), state = self._varCell(x_, state)
                mu = tf.tensordot(hidde_, Wdec_mu, [[-1], [0]]) + bdec_mu
                std = tf.nn.softplus(tf.tensordot(hidde_, Wdec_sig, [[-1], [0]]) + bdec_sig) + 1e-8
                x_ = tf.distributions.Normal(loc=mu, scale=std).sample()
                samples.append(x_)
            samples = tf.concat(samples, 0)
        return self._sess.run(samples)

    """#########################################################################
    output_function: reconstruction function.
    input: input - .
           samples - indicate wether outputs a sample.
    output: should be the reconstruction represented by the probability or the
            sample.
    #########################################################################"""
    def output_function(self, input, samples=True):
        with self._graph.as_default():
            if samples:
                return self._sess.run(tf.distributions.Normal(loc=self._dec[0], scale=self._dec[1]).sample(),
                                      feed_dict={self.x: input})
            else:
                mean, std = self._sess.run(self._dec, feed_dict={self.x: input})
                return mean, std
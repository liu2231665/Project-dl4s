"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the file contains the model description of STORN.
              ----2017.11.03
#########################################################################"""

import tensorflow as tf
from .utility import buildSTORN
from . import GaussKL, BernoulliNLL, GaussNLL
from dl4s.tools import get_batches_idx
import numpy as np
import time, os

"""#########################################################################
Class: _STORN - the hyper abstraction of the STORN.
#########################################################################"""
class _STORN(object):
    """#########################################################################
    __init__:the initialization function.
    input: Config - configuration class in ./utility.
    output: None.
    #########################################################################"""
    def __init__(
            self,
            configSTORN
    ):
        # Check the dimension configuration.
        if configSTORN.dimGen == []:
            raise (ValueError('The generating structure is empty!'))
        if configSTORN.dimReg == []:
            raise (ValueError('The recognition structure is empty!'))

        # <tensor graph> define a default graph.
        self._graph = tf.Graph()
        with self._graph.as_default():
            # <tensor placeholder> input.
            self.x = tf.placeholder(dtype='float32', shape=[None, None, configSTORN.dimInput])
            # <tensor placeholder> learning rate.
            self.lr = tf.placeholder(dtype='float32', shape=(), name='learningRate')
            # <scalar list> dimensions of hidden layers in generating model.
            self._dimGen = configSTORN.dimGen
            # <scalar list> dimensions of hidden layers in recognition model.
            self._dimReg = configSTORN.dimReg
            # <scalar> dimensions of input frame.
            self._dimInput = configSTORN.dimInput
            # <scalar> dimensions of stochastic states.
            self._dimState = configSTORN.dimState
            # <string/None> path to save the model.
            self._savePath = configSTORN.savePath
            # <string/None> path to save the events.
            self._eventPath = configSTORN.eventPath
            # <string/None> path to load the events.
            self._loadPath = configSTORN.loadPath
            # <list> collection of trainable parameters.
            self._params = []

            # build the structure.
            # self._muZ - the mean value of conditional Gaussian P(Z|X)         ###
            # self._sigZ - the std value of conditional Gaussian P(Z|X)    ###
            # self._hg_t - the hidden output of  value of the generating model, ###
            #              with Zt sampled from P(Z|X)                          ###
            # self._hiddenGen_t - the hidden output of  value of the generating ###
            #               model, with Zt sampled from prior P(Z)= N(0, 1)     ###
            # self._allCell - recurrent cell representing the whole model       ###
            # self._halfCell - recurrent cell representing the generating model.###
            self._muZ, self._sigZ, self._hg_t, self._half_hg_t, self._allCell, self._halfCell \
                = buildSTORN(self.x, self._graph, configSTORN)
            # <pass> will be define in the children classes.
            self._loss = GaussKL(self._muZ, self._sigZ**2, 0.0, 1.0)
            self._kl_divergence = self._loss
            # <pass> will be define in the children classes.
            self._train_step = None
            # <pass> the output of the recognition model.
            self._regOut = [self._muZ, self._sigZ]
            # <pass> the output of P(X|Z) given Z ~ P(Z|X) will be define in the children classes.
            self._allgenOut = None
            # <pass> the output of P(X|Z) given Z ~ N(0,1) will be define in the children classes.
            self._halfgenOut = None

            # <Tensorflow Optimizer>.
            if configSTORN.Opt == 'Adadelta':
                self._optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
            elif configSTORN.Opt == 'Adam':
                self._optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            elif configSTORN.Opt == 'Momentum':
                self._optimizer = tf.train.MomentumOptimizer(self.lr, 0.9)
            elif configSTORN.Opt == 'SGD':
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
    recognitionOutput: compute the P(Z|X) with given X.
    input: input - numerical input.
    output: the mean and std of P(Z|X).
    #########################################################################"""
    def recognitionOutput(self, input):
        with self._graph.as_default():
            zero_padd = np.zeros(shape=(input.shape[0], 1, input.shape[2]), dtype='float32')
            con_input = np.concatenate((zero_padd, input), axis=1)
            output = self._sess.run(self._regOut, feed_dict={self.x: con_input})
            return output[0][:, 0:-1, :], output[1][:, 0:-1, :]

    """#########################################################################
    output_function: reconstruction function.
    input: input - .
    output: should be the reconstruction represented by the probability.
    #########################################################################"""
    def output_function(self, input):
        with self._graph.as_default():
            zero_padd = np.zeros(shape=(input.shape[0], 1, input.shape[2]), dtype='float32')
            con_input = np.concatenate((zero_padd, input), axis=1)
            return self._sess.run(self._allgenOut, feed_dict={self.x: con_input})

"""#########################################################################
Class: binSTORN - the STORN model for stochastic binary inputs.
#########################################################################"""
class binSTORN(_STORN, object):
    """#########################################################################
    __init__:the initialization function.
    input: Config - configuration class in ./utility.
    output: None.
    #########################################################################"""
    def __init__(
            self,
            configSTORN
    ):
        super().__init__(configSTORN)
        with self._graph.as_default():
            with tf.variable_scope('logit'):
                W = tf.get_variable('W', shape=(configSTORN.dimGen[-1], configSTORN.dimInput))
                b = tf.get_variable('b', shape=configSTORN.dimInput, initializer=tf.zeros_initializer)
            # compute the generating outputs.
            self._allgenOut = tf.nn.sigmoid(tf.tensordot(self._hg_t, W, [[-1], [0]]) + b)
            self._halfgenOut = tf.nn.sigmoid(tf.tensordot(self._half_hg_t, W, [[-1], [0]]) + b)
            self._loss += BernoulliNLL(self.x[:, 1:, :], self._allgenOut)
            self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self._train_step = self._optimizer.minimize(tf.cast(tf.shape(self.x), tf.float32)[-1] * self._loss)
            self._runSession()

    """#########################################################################
    gen_function: generate samples.
    input: numSteps - the length of the sample sequence.
    output: should be the sample.
    #########################################################################"""
    def gen_function(self,  numSteps):
        with self._graph.as_default():
            state = self._halfCell.zero_state(1, dtype=tf.float32)
            x_ = tf.zeros((1, self._dimInput), dtype='float32')
            samples = []
            with tf.variable_scope('logit', reuse=True):
                W = tf.get_variable('W')
                b = tf.get_variable('b')
                for i in range(numSteps):
                    hidde_, state = self._halfCell(x_, state)
                    probs = tf.nn.sigmoid(tf.nn.xw_plus_b(hidde_, W, b))
                    x_ = tf.distributions.Bernoulli(probs=probs, dtype=tf.float32).sample()
                    samples.append(x_)
            samples = tf.concat(samples, 0)
        return self._sess.run(samples)

    """#########################################################################
    output_function: generate the reconstruction of input.
    input: input - .
    output: the reconstruction indicated by the binary probability.
    #########################################################################"""
    def output_function(self, input):
        with self._graph.as_default():
            zero_padd = np.zeros(shape=(input.shape[0], 1, input.shape[2]), dtype='float32')
            con_input = np.concatenate((zero_padd, input), axis=1)
            prob = self._sess.run( self._allgenOut, feed_dict={self.x: con_input})
        return prob


"""#########################################################################
Class: gaussSTORN - the STORN model for stochastic continuous inputs.
#########################################################################"""
class gaussSTORN(_STORN, object):
    """#########################################################################
    __init__:the initialization function.
    input: Config - configuration class in ./utility.
    output: None.
    #########################################################################"""
    def __init__(
            self,
            configSTORN
    ):
        super().__init__(configSTORN)
        with self._graph.as_default():
            with tf.variable_scope('output'):
                Wg_mu = tf.get_variable('Wg_mu', shape=(configSTORN.dimGen[-1], configSTORN.dimInput))
                bg_mu = tf.get_variable('bg_mu', shape=configSTORN.dimInput, initializer=tf.zeros_initializer)
                Wg_sig = tf.get_variable('Wg_sig', shape=(configSTORN.dimGen[-1], configSTORN.dimInput))
                bg_sig = tf.get_variable('bg_sig', shape=configSTORN.dimInput, initializer=tf.zeros_initializer)
            # compute the generating outputs.
            meanAll = tf.tensordot(self._hg_t, Wg_mu, [[-1], [0]]) + bg_mu
            sigALL = tf.nn.softplus(tf.tensordot(self._hg_t, Wg_sig, [[-1], [0]]) + bg_sig) + 1e-8
            self._allgenOut = [meanAll, sigALL]
            #
            meanHalf = tf.tensordot(self._half_hg_t, Wg_mu, [[-1], [0]]) + bg_mu
            sigHalf = tf.nn.softplus(tf.tensordot(self._half_hg_t, Wg_sig, [[-1], [0]]) + bg_sig) + 1e-8
            self._halfgenOut = [meanHalf, sigHalf]
            #
            # Compute the gaussian negative ll.
            self._loss += GaussNLL(self.x[:, 1:, :], self._allgenOut[0], self._allgenOut[1]**2)
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
            state = self._halfCell.zero_state(1, dtype=tf.float32)
            x_ = tf.zeros((1, self._dimInput), dtype='float32')
            samples = []
            with tf.variable_scope('output', reuse=True):
                Wg_mu = tf.get_variable('Wg_mu')
                bg_mu = tf.get_variable('bg_mu')
                Wg_sig = tf.get_variable('Wg_sig')
                bg_sig = tf.get_variable('bg_sig')
                for i in range(numSteps):
                    hidde_, state = self._halfCell(x_, state)
                    mu = tf.nn.xw_plus_b(hidde_, Wg_mu, bg_mu)
                    sig = tf.nn.softplus(tf.nn.xw_plus_b(hidde_, Wg_sig, bg_sig)) + 1e-8
                    x_ = tf.distributions.Normal(loc=mu, scale=sig).sample()
                    samples.append(x_)
            samples = tf.concat(samples, 0)
        return self._sess.run(samples)

    """#########################################################################
    output_function: generate the reconstruction of input.
    input: input - .
    output: the reconstruction indicated by the mean of conditional Gaussian.
    #########################################################################"""
    def output_function(self, input):
        with self._graph.as_default():
            zero_padd = np.zeros(shape=(input.shape[0], 1, input.shape[2]), dtype='float32')
            con_input = np.concatenate((zero_padd, input), axis=1)
            mean, std = self._sess.run(self._allgenOut, feed_dict={self.x: con_input})
        return mean, std
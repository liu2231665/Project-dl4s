# TODO:
"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the file contains the model description of SRNN.
              ----2017.11.03
#########################################################################"""

import tensorflow as tf
from .utility import buildSTORN
from . import GaussKL, BernoulliNLL, GaussNLL
import numpy as np

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
            # self._sigZ - the variance value of conditional Gaussian P(Z|X)    ###
            # self._hg_t - the hidden output of  value of the generating model, ###
            #              with Zt sampled from P(Z|X)                          ###
            # self._hiddenGen_t - the hidden output of  value of the generating ###
            #               model, with Zt sampled from prior P(Z)= N(0, 1)     ###
            # self._allCell - recurrent cell representing the whole model       ###
            # self._halfCell - recurrent cell representing the generating model.###
            self._muZ, self._sigZ, self._hg_t, self._half_hg_t, self._allCell, self._halfCell \
                = buildSTORN(self.x, self._graph, configSTORN)
            # <pass> will be define in the children classes.
            self._loss = GaussKL(self._muZ, self._sigZ, 0.0, 1.0)
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
    output_function: compute the P(Z|X) with given X.
    input: input - numerical input.
    output: the mean and variance of P(Z|X).
    #########################################################################"""
    def recognitionOutput(self, input):
        with self._graph.as_default():
            zero_padd = np.zeros(shape=(input.shape[0], 1, input.shape[2]), dtype='float32')
            con_input = np.concatenate((zero_padd, input), axis=1)
            output = self._sess.run(self._regOut, feed_dict={self.x: con_input})
            return output[0][:, 0:-1, :], output[1][:, 0:-1, :]

    """#########################################################################
    gen_function: generate samples.
    input: numSteps - the length of the sample sequence.
    output: should be the sample.
    #########################################################################"""
    def gen_function(self,  numSteps):
        return



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
            W = tf.get_variable('W', shape=(configSTORN.dimGen[-1], configSTORN.dimInput))
            b = tf.get_variable('b', shape=configSTORN.dimInput, initializer=tf.zeros_initializer)
            # compute the generating outputs.
            self._allgenOut = tf.nn.sigmoid(tf.tensordot(self._hg_t, W, [[-1], [0]]) + b)
            self._halfgenOut = tf.nn.sigmoid(tf.tensordot(self._half_hg_t, W, [[-1], [0]]) + b)
            self._loss -= BernoulliNLL(self.x[:, 1:, :], self._allgenOut)
            self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self._train_step = self._optimizer.minimize(tf.cast(tf.shape(self.x), tf.float32)[-1] * self._loss)
            self._runSession()

    """#########################################################################
    gen_function: generate samples.
    input: numSteps - the length of the sample sequence.
    output: should be the sample.
    #########################################################################"""
    def gen_function(self,  numSteps):
        X = np.zeros((1, numSteps+1, self._dimInput))
        prob = self._sess.run(self._halfgenOut, feed_dict={self.x: X})
        sample = np.random.binomial(1, prob[0, :, :])
        return sample


"""#########################################################################
Class: gaussSTORN - the STORN model for stochastic continuous inputs.
#########################################################################"""
class gaussSTORN(_STORN, object):
    pass
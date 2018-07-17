"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the file contains the model description of STORN.
              ----2017.11.03
#########################################################################"""

import tensorflow as tf
from .utility import buildSTORN
from dl4s.cores.tools import GaussKL, BernoulliNLL, GaussNLL
from dl4s.cores.model import _model
import numpy as np

"""#########################################################################
Class: _STORN - the hyper abstraction of the STORN.
#########################################################################"""
class _STORN(_model, object):
    """#########################################################################
    __init__:the initialization function.
    input: Config - configuration class in ./utility.
    output: None.
    #########################################################################"""
    def __init__(
            self,
            config
    ):
        # Check the dimension configuration.
        if config.dimGen == []:
            raise (ValueError('The generating structure is empty!'))
        if config.dimReg == []:
            raise (ValueError('The recognition structure is empty!'))
        _model.__init__(self, config=config)
        with self._graph.as_default():
            # <scalar list> dimensions of hidden layers in generating model.
            self._dimGen = config.dimGen
            # <scalar list> dimensions of hidden layers in recognition model.
            self._dimReg = config.dimReg
            # <scalar> dimensions of input frame.
            self._dimInput = config.dimIN
            # <scalar> dimensions of stochastic states.
            self._dimState = config.dimState

            # build the structure.
            # self._muZ - the mean value of conditional Gaussian P(Z|X)         ###
            # self._sigZ - the std value of conditional Gaussian P(Z|X)    ###
            # self._hg_t - the hidden output of  value of the generating model, ###
            #              with Zt sampled from P(Z|X)                          ###
            # self._hiddenGen_t - the hidden output of  value of the generating ###
            #               model, with Zt sampled from prior P(Z)= N(0, 1)     ###
            # self._allCell - recurrent cell representing the whole model       ###
            # self._halfCell - recurrent cell representing the generating model.###
            self._muZ, self._sigZ, self._hg_t, self._Cell = buildSTORN(self.x, self._graph, config)
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

    """#########################################################################
    encoder: compute the P(Z|X) with given X.
    input: input - numerical input.
    output: the mean and std of P(Z|X).
    #########################################################################"""
    def encoder(self, input):
        with self._graph.as_default():
            zero_padd = np.zeros(shape=(input.shape[0], 1, input.shape[2]), dtype='float32')
            con_input = np.concatenate((zero_padd, input), axis=1)
            output = self._sess.run(self._regOut, feed_dict={self.x: con_input})
            return output[0][:, 0:-1, :], output[1][:, 0:-1, :]


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
                W = tf.get_variable('W', shape=(configSTORN.dimGen[-1], self._dimInput))
                b = tf.get_variable('b', shape=self._dimInput, initializer=tf.zeros_initializer)
            # compute the generating outputs.
            self._dec = tf.nn.sigmoid(tf.tensordot(self._hg_t, W, [[-1], [0]]) + b)
            self._outputs = self._dec
            #
            self._loss += BernoulliNLL(self.x[:, 1:, :], self._outputs)
            self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self._train_step = self._optimizer.minimize(tf.cast(tf.shape(self.x), tf.float32)[-1] * self._loss)
            """define the process to generate samples."""
            # the initial state and initial input of the RNN.
            state = self._Cell.zero_state(1, dtype=tf.float32)
            x_ = tf.zeros((1, self._dimInput), dtype='float32')
            # TensorArray to save the output of the generating.
            gen_operator = tf.TensorArray(tf.float32, self.sampleLen)
            # condition and body of while loop (input: i-iteration, xx-RNN input, ss-RNN state)
            i = tf.constant(0)
            cond = lambda i, xx, ss, array: tf.less(i, self.sampleLen)
            #
            # Set the variational cell to use the prior P(Z) to generate Zt.
            self._Cell.setGen()

            def body(i, xx, ss, array):
                ii = i + 1
                (_, _, hidde_), new_ss = self._Cell(xx, ss)
                probs = tf.nn.sigmoid(tf.tensordot(hidde_, W, [[-1], [0]]) + b)
                new_xx = tf.distributions.Bernoulli(probs=probs, dtype=tf.float32).sample()
                new_array = array.write(i, new_xx)
                return ii, new_xx, new_ss, new_array

            gen_operator = tf.while_loop(cond, body, [i, x_, state, gen_operator])[-1]
            self._gen_operator = gen_operator.concat()
            #
            self._runSession()



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
                Wg_mu = tf.get_variable('Wg_mu', shape=(configSTORN.dimGen[-1], self._dimInput))
                bg_mu = tf.get_variable('bg_mu', shape=self._dimInput, initializer=tf.zeros_initializer)
                Wg_sig = tf.get_variable('Wg_sig', shape=(configSTORN.dimGen[-1], self._dimInput))
                bg_sig = tf.get_variable('bg_sig', shape=self._dimInput, initializer=tf.zeros_initializer)
            # compute the generating outputs.
            mu = tf.tensordot(self._hg_t, Wg_mu, [[-1], [0]]) + bg_mu
            std = tf.nn.softplus(tf.tensordot(self._hg_t, Wg_sig, [[-1], [0]]) + bg_sig) + 1e-8
            self._dec = [mu, std]
            self._outputs = mu
            #
            # Compute the gaussian negative ll.
            self._loss += GaussNLL(self.x[:, 1:, :], mu, std**2)
            self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self._train_step = self._optimizer.minimize(tf.cast(tf.shape(self.x), tf.float32)[-1] * self._loss)
            # TODO: define iteration for reconstruction.
            """define the process to generate samples."""
            # the initial state and initial input of the RNN.
            state = self._Cell.zero_state(1, dtype=tf.float32)
            x_ = tf.zeros((1, self._dimInput), dtype='float32')
            # TensorArray to save the output of the generating.
            gen_operator = tf.TensorArray(tf.float32, self.sampleLen)
            # condition and body of while loop (input: i-iteration, xx-RNN input, ss-RNN state)
            i = tf.constant(0)
            cond = lambda i, xx, ss, array: tf.less(i, self.sampleLen)
            #
            # Set the variational cell to use the prior P(Z) to generate Zt.
            self._Cell.setGen()

            def body(i, xx, ss, array):
                ii = i + 1
                (_, _, hidde_), new_ss = self._Cell(xx, ss)
                mu = tf.tensordot(hidde_, Wg_mu, [[-1], [0]]) + bg_mu
                sig = tf.nn.softplus(tf.tensordot(hidde_, Wg_sig, [[-1], [0]]) + bg_sig) + 1e-8
                new_xx = tf.distributions.Normal(loc=mu, scale=sig).sample()
                new_array = array.write(i, new_xx)
                return ii, new_xx, new_ss, new_array

            gen_operator = tf.while_loop(cond, body, [i, x_, state, gen_operator])[-1]
            self._gen_operator = gen_operator.concat()
            #
            self._runSession()
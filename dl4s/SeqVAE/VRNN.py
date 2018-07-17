"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the file contains the model description of VRNN.
              ----2017.11.13
#########################################################################"""

import tensorflow as tf
from .utility import buildVRNN
from .utility import configVRNN
from dl4s.cores.tools import GaussKL, BernoulliNLL, GaussNLL
from dl4s.cores.model import _model

"""#########################################################################
Class: _VRNN - the hyper abstraction of the VRNN.
#########################################################################"""
class _VRNN(_model, object):
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
        if config.dimRec == []:
            raise (ValueError('The recurrent structure is empty!'))
        _model.__init__(self, config=config)
        with self._graph.as_default():
            # <scalar list> the size of recurrent hidden layers.
            self._dimRec = config.dimRec
            # <scalar list> the size of feedforward hidden layers of input.
            self._dimForX = config.dimForX
            # <scalar list> the size of feedforward hidden layers of stochastic layer.
            self._dimForZ = config.dimForZ
            # <scalar> dimensions of input frame.
            self._dimInput = config.dimIN
            #
            self._prior_mu, self._prior_sig, self._pos_mu, self._pos_sig,\
            self._hidden_dec, self._varCell, self._Z = buildVRNN(self.x, self._graph, config)
            self._loss = GaussKL(self._pos_mu, self._pos_sig ** 2, self._prior_mu, self._prior_sig ** 2)
            self._kl_divergence = self._loss
            # the prior P(Z).
            self._prior = [self._prior_mu, self._prior_sig]
            # the posterior P(Z|X).
            self._enc = [self._pos_mu, self._pos_sig]
            # <pass> compute the posterior P(X|Z)
            self._dec = []
            # using the E(Z|X) as extracted feature.
            self._feature = self._pos_mu

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
                Wdec = tf.get_variable('Wdec', shape=(self._hidden_dec.shape[-1], self._dimInput))
                bdec = tf.get_variable('bdec', shape=self._dimInput, initializer=tf.zeros_initializer)
                self._dec = tf.nn.sigmoid(tf.tensordot(self._hidden_dec, Wdec, [[-1], [0]]) + bdec)
                self._outputs = self._dec
                #
                self._loss += BernoulliNLL(self.x, self._dec)
                self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                self._train_step = self._optimizer.minimize(tf.cast(tf.shape(self.x), tf.float32)[-1] * self._loss)
                """define the process to generate samples."""
                # the initial state and initial input of the RNN.
                state = self._varCell.zero_state(1, dtype=tf.float32)
                x_ = tf.zeros((1, self._dimInput), dtype='float32')
                # TensorArray to save the output of the generating.
                gen_operator = tf.TensorArray(tf.float32, self.sampleLen)
                # condition and body of while loop (input: i-iteration, xx-RNN input, ss-RNN state)
                i = tf.constant(0)
                cond = lambda i, xx, ss, array: tf.less(i, self.sampleLen)
                #
                # Set the variational cell to use the prior P(Z) to generate Zt.
                self._varCell.setGen()
                def body(i, xx, ss, array):
                    ii = i + 1
                    (_, _, _, _, hidde_, _, _), new_ss = self._varCell(xx, ss)
                    probs = tf.nn.sigmoid(tf.nn.xw_plus_b(hidde_, Wdec, bdec))
                    new_xx = tf.distributions.Bernoulli(probs=probs, dtype=tf.float32).sample()
                    new_array = array.write(i, new_xx)
                    return ii, new_xx, new_ss, new_array

                gen_operator = tf.while_loop(cond, body, [i, x_, state, gen_operator])[-1]
                self._gen_operator = gen_operator.concat()
                self._runSession()

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
                Wdec_mu = tf.get_variable('Wdec_mu', shape=(self._hidden_dec.shape[-1], self._dimInput))
                bdec_mu = tf.get_variable('bdec_mu', shape=self._dimInput, initializer=tf.zeros_initializer)
                mu = tf.tensordot(self._hidden_dec, Wdec_mu, [[-1], [0]]) + bdec_mu
                Wdec_sig = tf.get_variable('Wdec_sig', shape=(self._hidden_dec.shape[-1], self._dimInput))
                bdec_sig = tf.get_variable('bdec_sig', shape=self._dimInput, initializer=tf.zeros_initializer)
                std = tf.nn.softplus(tf.tensordot(self._hidden_dec, Wdec_sig, [[-1], [0]]) + bdec_sig) + 1e-8
                self._dec = [mu, std]
                self._outputs = mu
                #
                self._loss += GaussNLL(self.x, mu, std**2)
                self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                self._train_step = self._optimizer.minimize(tf.cast(tf.shape(self.x), tf.float32)[-1] * self._loss)
                """define the process to generate samples."""
                # the initial state and initial input of the RNN.
                state = self._varCell.zero_state(1, dtype=tf.float32)
                x_ = tf.zeros((1, self._dimInput), dtype='float32')
                # TensorArray to save the output of the generating.
                gen_operator = tf.TensorArray(tf.float32, self.sampleLen)
                # condition and body of while loop (input: i-iteration, xx-RNN input, ss-RNN state)
                i = tf.constant(0)
                cond = lambda i, xx, ss, array: tf.less(i, self.sampleLen)
                #
                # Set the variational cell to use the prior P(Z) to generate Zt.
                self._varCell.setGen()

                def body(i, xx, ss, array):
                    ii = i + 1
                    (_, _, _, _, hidde_, _, _), new_ss = self._varCell(xx, ss)
                    mu = tf.tensordot(hidde_, Wdec_mu, [[-1], [0]]) + bdec_mu
                    sig = tf.nn.softplus(tf.tensordot(hidde_, Wdec_mu, [[-1], [0]]) + bdec_mu) + 1e-8
                    new_xx = tf.distributions.Normal(loc=mu, scale=sig).sample()
                    new_array = array.write(i, new_xx)
                    return ii, new_xx, new_ss, new_array

                gen_operator = tf.while_loop(cond, body, [i, x_, state, gen_operator])[-1]
                self._gen_operator = gen_operator.concat()
                self._runSession()
"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the file contains the model description of SRNN.
              ----2017.11.15
#########################################################################"""
from dl4s.cores.tools import GaussKL, BernoulliNLL, GaussNLL
from dl4s.cores.model import _model
from dl4s.SeqVAE import configSRNN
from dl4s.SeqVAE.utility import buildSRNN
import tensorflow as tf

"""#########################################################################
Class: _SRNN - the hyper abstraction of the SRNN.
#########################################################################"""
class _SRNN(_model, object):
    """#########################################################################
    __init__:the initialization function.
    input: Config - configuration class in ./utility.
    output: None.
    #########################################################################"""
    def __init__(
            self,
            config
    ):
        # Check the froward recurrent dimension configuration.
        if config.dimRecD == []:
            raise (ValueError('The forward recurrent structure is empty!'))
        # Check the froward recurrent dimension configuration.
        if config.dimRecA == [] and config.mode == 'smooth':
            raise (ValueError('The backward recurrent structure in smooth mode is empty!'))
        _model.__init__(self, config=config)
        with self._graph.as_default():
            # <scalar> dimensions of input frame.
            self._dimInput = config.dimIN
            # <scalar> dimensions of stochastic states.
            self._dimState = config.dimState
            # <scalar list> the size of forward recurrent hidden layers.
            self._dimRecD = config.dimRecD
            # <scalar list> the size of backward recurrent hidden layers/MLP depended on the mode.
            self._dimRecA = config.dimRecA
            # components of the SRNN.
            self._prior_mu, self._prior_sig, self._pos_mu, self._pos_sig, self._hidden_dec, \
            [self._forwardCell, self._SSM, self._MLPx], self._Z = buildSRNN(self.x, self._graph, config)
            # the loss functions.
            self._loss = GaussKL(self._pos_mu, self._pos_sig ** 2, self._prior_mu, self._prior_sig ** 2)
            self._kl_divergence = self._loss
            # the prior P(Z).
            self._prior = [self._prior_mu, self._prior_sig]
            # the posterior P(Z|X).
            self._enc = [self._pos_mu, self._pos_sig]
            # using the E(Z|X) as extracted feature.
            self._feature = self._pos_mu
            # <pass> compute the posterior P(X|Z)
            self._dec = []

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
Class: binSRNN - the SRNN model for stochastic binary inputs..
#########################################################################"""
class binSRNN(_SRNN, object):
    """#########################################################################
    __init__:the initialization function.
    input: Config - configuration class in ./utility.
    output: None.
    #########################################################################"""
    def __init__(
            self,
            config=configSRNN
    ):
        super().__init__(config)
        with self._graph.as_default():
            with tf.variable_scope('logit'):
                Wdec = tf.get_variable('Wdec', shape=(self._hidden_dec.shape[-1], config.dimIN))
                bdec = tf.get_variable('bdec', shape=config.dimIN, initializer=tf.zeros_initializer)
                self._dec = tf.nn.sigmoid(tf.tensordot(self._hidden_dec, Wdec, [[-1], [0]]) + bdec)
                self._outputs = self._dec
                self._loss += BernoulliNLL(self.x, self._dec)
                self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                self._train_step = self._optimizer.minimize(tf.cast(tf.shape(self.x), tf.float32)[-1] * self._loss)
                """define the process to generate samples."""
                # the initial state and initial input of the RNN.
                state0 = self._forwardCell.zero_state(1, dtype=tf.float32)
                state1 = self._SSM.zero_state(1, dtype=tf.float32)
                x_ = tf.zeros((1, self._dimInput), dtype=tf.float32)
                # TensorArray to save the output of the generating.
                gen_operator = tf.TensorArray(tf.float32, self.sampleLen)
                # condition and body of while loop (input: i-iteration, xx-RNN input, ss-RNN state)
                i = tf.constant(0)
                cond = lambda i, xx, ss0, ss1, array: tf.less(i, self.sampleLen)
                #
                # Set the variational cell to use the prior P(Z) to generate Zt.
                self._SSM.setGen()

                def body(i, xx, ss0, ss1, array):
                    ii = i + 1
                    d_t, new_ss0 = self._forwardCell(self._MLPx(xx), ss0)
                    a_t = tf.zeros((1, self._dimRecA[-1]), dtype=tf.float32)
                    input = tf.concat(axis=-1, values=(d_t, a_t))
                    (_, _, _, _, hidden_dec, _), new_ss1 = self._SSM(input, ss1)
                    probs = tf.nn.sigmoid(tf.tensordot(hidden_dec, Wdec, [[-1], [0]]) + bdec)
                    new_xx = tf.distributions.Bernoulli(probs=probs, dtype=tf.float32).sample()
                    new_array = array.write(i, new_xx)
                    return ii, new_xx, new_ss0, tf.reshape(new_ss1, shape=(1, self._dimState)), new_array

                gen_operator = tf.while_loop(cond, body, [i, x_, state0, state1, gen_operator])[-1]
                self._gen_operator = gen_operator.concat()
                self._runSession()

"""#########################################################################
Class: gaussSRNN - the SRNN model for stochastic continuous inputs.
#########################################################################"""
class gaussSRNN(_SRNN, object):
    """#########################################################################
    __init__:the initialization function.
    input: Config - configuration class in ./utility.
    output: None.
    #########################################################################"""
    def __init__(
            self,
            config=configSRNN
    ):
        super().__init__(config)
        with self._graph.as_default():
            with tf.variable_scope('output'):
                # compute the mean and standard deviation of P(X|Z).
                Wdec_mu = tf.get_variable('Wdec_mu', shape=(self._hidden_dec.shape[-1], config.dimIN))
                bdec_mu = tf.get_variable('bdec_mu', shape=config.dimIN, initializer=tf.zeros_initializer)
                mu = tf.tensordot(self._hidden_dec, Wdec_mu, [[-1], [0]]) + bdec_mu
                Wdec_sig = tf.get_variable('Wdec_sig', shape=(self._hidden_dec.shape[-1], config.dimIN))
                bdec_sig = tf.get_variable('bdec_sig', shape=config.dimIN, initializer=tf.zeros_initializer)
                std = tf.nn.softplus(tf.tensordot(self._hidden_dec, Wdec_sig, [[-1], [0]]) + bdec_sig) + 1e-8
                self._dec = [mu, std]
                self._outputs = mu

                self._loss += GaussNLL(self.x, mu, std ** 2)
                self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                self._train_step = self._optimizer.minimize(tf.cast(tf.shape(self.x), tf.float32)[-1] * self._loss)
                """define the process to generate samples."""
                # the initial state and initial input of the RNN.
                state0 = self._forwardCell.zero_state(1, dtype=tf.float32)
                state1 = self._SSM.zero_state(1, dtype=tf.float32)
                x_ = tf.zeros((1, self._dimInput), dtype=tf.float32)
                # TensorArray to save the output of the generating.
                gen_operator = tf.TensorArray(tf.float32, self.sampleLen)
                # condition and body of while loop (input: i-iteration, xx-RNN input, ss-RNN state)
                i = tf.constant(0)
                cond = lambda i, xx, ss0, ss1, array: tf.less(i, self.sampleLen)
                #
                # Set the variational cell to use the prior P(Z) to generate Zt.
                self._SSM.setGen()

                def body(i, xx, ss0, ss1, array):
                    ii = i + 1
                    d_t, new_ss0 = self._forwardCell(self._MLPx(xx), ss0)
                    a_t = tf.zeros((1, self._dimRecA[-1]), dtype=tf.float32)
                    input = tf.concat(axis=-1, values=(d_t, a_t))
                    (_, _, _, _, hidden_dec, _), new_ss1 = self._SSM(input, ss1)
                    mu = tf.tensordot(hidden_dec, Wdec_mu, [[-1], [0]]) + bdec_mu
                    std = tf.nn.softplus(tf.tensordot(hidden_dec, Wdec_sig, [[-1], [0]]) + bdec_sig) + 1e-8
                    new_xx = tf.distributions.Normal(loc=mu, scale=std).sample()
                    new_array = array.write(i, new_xx)
                    return ii, new_xx, new_ss0, tf.reshape(new_ss1, shape=(1, self._dimState)), new_array

                gen_operator = tf.while_loop(cond, body, [i, x_, state0, state1, gen_operator])[-1]
                self._gen_operator = gen_operator.concat()
                self._runSession()

"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the file contains the model description of CGRNN.
              ----2017.11.15
#########################################################################"""
from .utility import configCGRNN, CGCell
from dl4s.cores.tools import BernoulliNLL
from dl4s.cores.model import _model
import tensorflow as tf
import numpy as np

"""#########################################################################
Class: _CGRNN - the hyper abstraction of the CGRNN.
#########################################################################"""
class _CGRNN(_model, object):
    """#########################################################################
    __init__:the initialization function.
    input: Config - configuration class in ./utility.
    output: None.
    #########################################################################"""
    def __init__(
            self,
            config=configCGRNN()
    ):
        # Check the froward recurrent dimension configuration.
        if config.dimRec == []:
            raise (ValueError('The forward recurrent structure is empty!'))

        super().__init__(config)
        with self._graph.as_default():
            # <scalar> the steps of Gibbs sampling.
            self._gibbs = config.Gibbs
            # <scalar> the number of samples of AIS.
            self._aisRun = config.aisRun
            # <scalar> the number of intermediate proposal distributions of AIS.
            self._aisLevel = config.aisLevel
            # <scalar> dimensions of input frame.
            self._dimInput = config.dimIN
            # <scalar> dimensions of stochastic states.
            self._dimState = config.dimState
            # <scalar list> the size of forward recurrent hidden layers.
            self._dimRec = config.dimRec
            # <scalar list> the size of feed-forward hidden layers.
            self._dimMlp = config.dimMlp
            # <string> the mode.
            self._mode = config.mode
            self.VAE = None

"""#########################################################################
Class: binCGRNN - the CGRNN mode for binary input.
#########################################################################"""
class binCGRNN(_CGRNN, object):
    """#########################################################################
    __init__:the initialization function.
    input: Config - configuration class in ./utility.
           VAE - if a well trained VAE is provided. Using NVIL to estimate the
                 upper bound of the partition function.
    output: None.
    #########################################################################"""
    def __init__(
            self,
            config=configCGRNN(),
            VAE=None
    ):
        super().__init__(config)
        """build the graph"""
        with self._graph.as_default():
            self.Cell = CGCell(config, inputType='binary')
            state = self.Cell.zero_state(tf.shape(self.x)[0], dtype=tf.float32)
            (self.newV, self.newH, self.newS, self.muV, self.muH, self.muS, bvt, bht), _ = \
                tf.nn.dynamic_rnn(self.Cell, self.x, initial_state=state)
            # update the RBM's bias with bvt & bht.
            self.Cell.RBM._bh = bht
            self.Cell.RBM._bv = bvt
            # one step sample.
            muV0, muH0, muS0 = self.Cell.RBM.GibbsSampling(self.x, k=1)[-3:]
            # add the tensor computation of extracted feature.
            self._outputs = muV0
            self._feature = muH0
            self._sparse_feature = muH0 * muS0
            # the training loss is per bits.
            Loss = self.Cell.RBM.ComputeLoss(V=self.x, samplesteps=config.Gibbs)
            self._loss = BernoulliNLL(self.x, self.muV)
            self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self._train_step = self._optimizer.minimize(Loss)
            # Define the components to evaluate the partition function by whether NVIL or AIS.
            if VAE is None:
                self._logZ = self.Cell.RBM.AIS(self._aisRun, self._aisLevel,
                                           tf.shape(self.x)[0], tf.shape(self.x)[1])
                self._nll = tf.reduce_mean(self.Cell.RBM.FreeEnergy(self.x) + self._logZ)
                #self._nll = self._logZ
                #self._nll = self.Cell.RBM.FreeEnergy(self.x)
                self.VAE = VAE
            else:
                self._logZ = self._NVIL_VAE(VAE)  # X, logPz_X, logPx_Z, logPz, VAE.x
                self.xx = tf.placeholder(dtype='float32', shape=[None, None, None, config.dimIN])
                self.FEofSample = self.Cell.RBM.FreeEnergy(self.xx)
                self.FEofInput = self.Cell.RBM.FreeEnergy(self.x)
                self.VAE = VAE
            """define the process to generate samples."""
            state = self.Cell.zero_state(1, dtype=tf.float32)
            x_ = tf.zeros((1, self._dimInput), dtype='float32')
            # TensorArray to save the output of the generating.
            gen_operator = tf.TensorArray(tf.float32, self.sampleLen)
            # condition and body of while loop (input: i-iteration, xx-RNN input, ss-RNN state)
            i = tf.constant(0)
            cond = lambda i, xx, ss, array: tf.less(i, self.sampleLen)
            #
            def body(i, xx, ss, array):
                ii = i + 1
                (new_xx, _, _, _, _, _, _, _), new_ss = self.Cell(xx, ss, gibbs=1)
                new_array = array.write(i, new_xx)
                return ii, new_xx, new_ss, new_array

            gen_operator = tf.while_loop(cond, body, [i, x_, state, gen_operator])[-1]
            self._gen_operator = gen_operator.concat()
            #
            self._runSession()

    """#########################################################################
    _NVIL_VAE: generate the graph to compute the NVIL upper bound of log Partition
               function by a well-trained VAE.
    input: VAE - the well-trained VAE(SRNN/VRNN).
    output: the upper boundLogZ.
    #########################################################################"""
    def _NVIL_VAE(self, VAE):
        # get the marginal and conditional distribution of the VAE.
        probs = VAE._dec
        Px_Z = tf.distributions.Bernoulli(probs=probs, dtype=tf.float32)
        mu, std = VAE._enc
        Pz_X = tf.distributions.Normal(loc=mu, scale=std)
        mu, std = VAE._prior
        Pz = tf.distributions.Normal(loc=mu, scale=std)
        # generate the samples.
        X = Px_Z.sample()
        logPz_X = tf.reduce_sum(Pz_X.log_prob(VAE._Z), axis=[-1])  # shape = [batch, steps]
        # logPx_Z = tf.reduce_prod(Px_Z.log_prob(X), axis=[-1])
        logPx_Z = tf.reduce_sum(
            (1 - X) * tf.log(tf.maximum(tf.minimum(1.0, 1 - probs), 1e-32))
            + X * tf.log(tf.maximum(tf.minimum(1.0, probs), 1e-32)),
            axis=[-1])  # shape = [runs, batch, steps]
        logPz = tf.reduce_sum(Pz.log_prob(VAE._Z), axis=[-1])
        return X, logPz_X, logPx_Z, logPz, VAE.x

    """#########################################################################
    ais_function: compute the approximated negative log-likelihood with partition
                  function computed by annealed importance sampling.
    input: input - numerical input.
    output: the negative log-likelihood value.
    #########################################################################"""
    def ais_function(self, input):
        with self._graph.as_default():
            if self.VAE is None:
                loss_value = self._sess.run(self._nll, feed_dict={self.x: input})
            else:
                loss_value = []
                X = []
                logPz_X = []
                logPx_Z = []
                logPz = []
                for i in range(self._aisRun):
                    Xi, logPz_Xi, logPx_Zi, logPzi = self.VAE._sess.run(self._logZ[0:-1], feed_dict={self._logZ[-1]: input})
                    X.append(Xi)
                    logPz_X.append(logPz_Xi)
                    logPx_Z.append(np.nan_to_num(logPx_Zi))
                    logPz.append(logPzi)
                    # shape = [runs, batch, steps]
                X = np.asarray(X, dtype=np.float64)
                logPz_X = np.asarray(logPz_X, dtype=np.float64)
                logPx_Z = np.asarray(logPx_Z, dtype=np.float64)
                logPz = np.asarray(logPz, dtype=np.float64)
                FEofSample = self._sess.run(self.FEofSample, feed_dict={self.xx: X, self.x: input})
                FEofSample = np.cast[np.float64](FEofSample)
                logTerm = 2 * (-FEofSample + logPz_X - logPx_Z - logPz) / 1000 #self._dimInput
                r_ais = np.mean(np.exp(logTerm), axis=0)
                logZ = 0.5 * (np.log(r_ais+1e-38))
                FEofInput = self._sess.run(self.FEofInput, feed_dict={self.x: input})
                FEofInput = np.cast[np.float64](FEofInput)
                loss_value.append(np.mean(FEofInput + logZ * 1000))#self._dimInput))
                loss_value = np.asarray(loss_value).mean()
        return loss_value

# TODO"
"""#########################################################################
Class: gaussCGRNN - the CGRNN mode for continuous input.
#########################################################################"""
class gaussCGRNN(_CGRNN, object):
    """#########################################################################
    __init__:the initialization function.
    input: Config - configuration class in ./utility.
           VAE - if a well trained VAE is provided. Using NVIL to estimate the
                 upper bound of the partition function.
    output: None.
    #########################################################################"""
    def __init__(
            self,
            config,
            VAE=None
    ):
        super().__init__(config)
        """build the graph"""
        with self._graph.as_default():
            self.Cell = CGCell(config, inputType='continuous')
            state = self.Cell.zero_state(tf.shape(self.x)[0], dtype=tf.float32)
            (self.newV, self.newH, self.newS, self.muV, self.muH, self.muS,
             self.bvt, self.bht, self.gamma), _ = tf.nn.dynamic_rnn(self.Cell, self.x, initial_state=state)
            # update the RBM's bias with bvt & bht, gamma.
            self.Cell.RBM._bh = self.bht
            self.Cell.RBM._bv = self.bvt
            self.Cell.RBM._gamma = self.gamma
            # one step sample.
            muV0, muH0, muS0 = self.Cell.RBM.GibbsSampling(self.x, k=1)[-3:]
            # add the tensor computation of extracted feature.
            self._outputs = muV0
            self._feature = muH0
            self._sparse_feature = muH0 * muS0
            # the training loss is per frame.
            Loss = self.Cell.RBM.ComputeLoss(V=self.x, samplesteps=config.Gibbs)
            # define the monitor.
            monitor = tf.reduce_sum((self.x - self.muV) ** 2, axis=-1)
            self._loss = tf.sqrt(tf.reduce_mean(monitor))
            self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self._train_step = self._optimizer.minimize(Loss)
            # add the computation of precision and covariance matrix of ssRBM.
            newH = tf.expand_dims(self.newH, axis=2)
            W = tf.expand_dims(tf.expand_dims(self.Cell.RBM._W, axis=0), axis=0)
            term1 = newH * W / (self.Cell.RBM._alpha + 1e-8)
            term1 = tf.tensordot(term1, self.Cell.RBM._W, [[-1], [-1]])
            Cv_sh = 1 / (tf.expand_dims(self.Cell.RBM._gamma, axis=2) + tf.tensordot(newH, self.Cell.RBM._phi, [[-1], [0]]) + 1e-8)
            term2 = Cv_sh * tf.eye(self._dimInput, batch_shape=[1, 1])
            self.PreV_h = term2 + term1
            self.CovV_h = tf.matrix_inverse(self.PreV_h)
            #
            if VAE is None:
                self._logZ = self.Cell.RBM.AIS(self._aisRun, self._aisLevel,
                                           tf.shape(self.x)[0], tf.shape(self.x)[1])
                self._nll = tf.reduce_mean(self.Cell.RBM.FreeEnergy(self.x) + self._logZ)
                self.VAE = VAE
            else:
                self._logZ = self._NVIL_VAE(VAE)  # X, logPz_X, logPx_Z, logPz, VAE.x
                self.xx = tf.placeholder(dtype='float32', shape=[None, None, None, config.dimIN])
                self.FEofSample = self.Cell.RBM.FreeEnergy(self.xx)
                self.FEofInput = self.Cell.RBM.FreeEnergy(self.x)
                self.VAE = VAE
            """define the process to generate samples."""
            state = self.Cell.zero_state(1, dtype=tf.float32)
            x_ = tf.zeros((1, self._dimInput), dtype='float32')
            # TensorArray to save the output of the generating.
            gen_operator = tf.TensorArray(tf.float32, self.sampleLen)
            # condition and body of while loop (input: i-iteration, xx-RNN input, ss-RNN state)
            i = tf.constant(0)
            cond = lambda i, xx, ss, array: tf.less(i, self.sampleLen)

            #
            def body(i, xx, ss, array):
                ii = i + 1
                (new_xx, _, _, _, _, _, _, _, _), new_ss = self.Cell(xx, ss, gibbs=1)
                new_array = array.write(i, new_xx)
                return ii, new_xx, new_ss, new_array

            gen_operator = tf.while_loop(cond, body, [i, x_, state, gen_operator])[-1]
            self._gen_operator = gen_operator.concat()
            #
            self._runSession()

    """#########################################################################
    ais_function: compute the approximated negative log-likelihood with partition
                  function computed by annealed importance sampling.
    input: input - numerical input.
    output: the negative log-likelihood value.
    #########################################################################"""
    def ais_function(self, input):
        with self._graph.as_default():
            if self.VAE is None:
                loss_value = self._sess.run(self._nll, feed_dict={self.x: input})
            else:
                loss_value = []
                X = []
                logPz_X = []
                logPx_Z = []
                logPz = []
                for i in range(self._aisRun):
                    Xi, logPz_Xi, logPx_Zi, logPzi = self.VAE._sess.run(self._logZ[0:-1],
                                                                        feed_dict={self._logZ[-1]: input})
                    X.append(Xi)
                    logPz_X.append(np.nan_to_num(logPz_Xi))
                    logPx_Z.append(np.nan_to_num(logPx_Zi))
                    logPz.append(np.nan_to_num(logPzi))
                    # shape = [runs, batch, steps]
                X = np.asarray(X, dtype=np.float64)
                logPz_X = np.asarray(logPz_X, dtype=np.float64)
                logPx_Z = np.asarray(logPx_Z, dtype=np.float64)
                logPz = np.asarray(logPz, dtype=np.float64)
                FEofSample = self._sess.run(self.FEofSample, feed_dict={self.xx: X, self.x: input})
                FEofSample = np.cast[np.float64](FEofSample)
                logTerm = 2 * (-FEofSample + logPz_X - logPx_Z - logPz) / 1000  # self._dimInput
                r_ais = np.mean(np.exp(logTerm), axis=0)
                logZ = 0.5 * (np.log(r_ais + 1e-38))
                FEofInput = self._sess.run(self.FEofInput, feed_dict={self.x: input})
                FEofInput = np.cast[np.float64](FEofInput)
                loss_value.append(np.mean(FEofInput + logZ * 1000))  # self._dimInput))
                loss_value = np.asarray(loss_value).mean()
        return loss_value

    """#########################################################################
    _NVIL_VAE: generate the graph to compute the NVIL upper bound of log Partition
               function by a well-trained VAE.
    input: VAE - the well-trained VAE(SRNN/VRNN).
    output: the upper boundLogZ.
    #########################################################################"""
    def _NVIL_VAE(self, VAE):
        # get the marginal and conditional distribution of the VAE.
        mu, std = VAE._dec
        Px_Z = tf.distributions.Normal(loc=mu, scale=std)
        mu1, std1 = VAE._enc
        Pz_X = tf.distributions.Normal(loc=mu1, scale=std1)
        mu, std = VAE._prior
        Pz = tf.distributions.Normal(loc=mu, scale=std)
        # generate the samples.
        X = Px_Z.sample()
        logPz_X = tf.reduce_sum(Pz_X.log_prob(VAE._Z), axis=[-1])  # shape = [batch, steps]
        logPx_Z = tf.reduce_sum(Px_Z.log_prob(X), axis=[-1])
        logPz = tf.reduce_sum(Pz.log_prob(VAE._Z), axis=[-1])
        return X, logPz_X, logPx_Z, logPz, VAE.x

    """#########################################################################
    cov_function: compute the covariance matrix Cv_h.
    input: input - numerical input.
    output:  covariance matrix Cv_h.
    #########################################################################"""
    def cov_function(self, input):
        with self._graph.as_default():
            return self._sess.run(self.CovV_h, feed_dict={self.x: input})

    """#########################################################################
    pre_function: compute the precision matrix Cv_h^{-1}.
    input: input - numerical input.
    output:  precision matrix Cv_h^{-1}.
    #########################################################################"""
    def pre_function(self, input):
        with self._graph.as_default():
            return self._sess.run(self.PreV_h, feed_dict={self.x: input})

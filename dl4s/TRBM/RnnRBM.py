"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the file contains the model description of RNN-RBM.
              ----2017.11.03
#########################################################################"""
from dl4s.TRBM import configRNNRBM, configssRNNRBM
from dl4s.SeqVAE.utility import buildRec, MLP
from dl4s.TRBM.RBM import binRBM, gaussRBM, mu_ssRBM, bin_ssRBM
from dl4s.cores.model import _model
import tensorflow as tf
import numpy as np

"""#########################################################################
Class: _RnnRBM - the hyper abstraction of the RnnRBM.
#########################################################################"""
class _RnnRBM(_model, object):
    """#########################################################################
    __init__:the initialization function.
    input: Config - configuration class in ./utility.
    output: None.
    #########################################################################"""
    def __init__(
            self,
            config=configRNNRBM()
    ):
        # Check the froward recurrent dimension configuration.
        if config.dimRec == []:
            raise (ValueError('The recurrent structure is empty!'))
        _model.__init__(self, config=config)
        with self._graph.as_default():
            # <scalar> the number of samples of AIS.
            self._aisRun = config.aisRun
            # <scalar> the number of intermediate proposal distributions of AIS.
            self._aisLevel = config.aisLevel
            # <scalar> the steps of Gibbs sampling.
            self._gibbs = config.Gibbs
            # <scalar> the size of frame of the input.
            self._dimInput = config.dimIN
            # <scalar> the size of frame of the state.
            self._dimState = config.dimState
            # <list> dims of recurrent layers.
            self._dimRec = config.dimRec
            # <list> the RNN components.
            self._rnnCell = buildRec(dimLayer=config.dimRec, unitType=config.recType,
                                     init_scale=config.init_scale)
            #
            #
            self._rbm = None
            self._nll = None

"""#########################################################################
Class: binRnnRBM - the RNNRBM model for stochastic binary inputs.
#########################################################################"""
class binRnnRBM(_RnnRBM, object):
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
            # d_t = [batch, steps, hidden]
            self._mlp = MLP(config.init_scale, config.dimIN, config.dimMlp, config.mlpType)
            state = self._rnnCell.zero_state(tf.shape(self.x)[0], dtype=tf.float32)
            d, _ = tf.nn.dynamic_rnn(self._rnnCell, self._mlp(self.x), initial_state=state)
            paddings = tf.constant([[0, 0], [1, 0], [0, 0]])
            dt = tf.pad(d[:, 0:-1, :], paddings)
            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
            with tf.variable_scope("RBM", initializer=initializer):
                bv = tf.get_variable('bv', shape=config.dimIN, initializer=tf.zeros_initializer)
                bh = tf.get_variable('bh', shape=config.dimState, initializer=tf.zeros_initializer)
                Wdv = tf.get_variable('Wdv', shape=[config.dimRec[-1], config.dimIN])
                Wdh = tf.get_variable('Wdh', shape=[config.dimRec[-1], config.dimState])
                bvt = tf.tensordot(dt, Wdv, [[-1], [0]]) + bv
                bht = tf.tensordot(dt, Wdh, [[-1], [0]]) + bh
            self._rbm = binRBM(dimV=config.dimIN, dimH=config.dimState, init_scale=config.init_scale,
                               x=self.x, bv=bvt, bh=bht, k=self._gibbs)
            # the training loss is per frame.
            Loss = self._rbm.ComputeLoss(V=self.x, samplesteps=self._gibbs)
            if VAE is None:
                # The component for computing AIS.
                self._logZ = self._rbm.AIS(self._aisRun, self._aisLevel,
                                           tf.shape(self.x)[0], tf.shape(self.x)[1])
                self._nll = tf.reduce_mean(self._rbm.FreeEnergy(self.x) + self._logZ)
                self.VAE = VAE
            else:
                # The component for computing NVIL.
                self._logZ = self._NVIL_VAE(VAE)  # X, logPz_X, logPx_Z, logPz, VAE.x
                self.xx = tf.placeholder(dtype='float32', shape=[None, None, None, config.dimIN])
                self.FEofSample = self._rbm.FreeEnergy(self.xx)
                self.FEofInput = self._rbm.FreeEnergy(self.x)
                self.VAE = VAE
            #
            self._loss = self._rbm._monitor
            self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self._train_step = self._optimizer.minimize(Loss)
            # Define the reconstruction of input.
            self._outputs = self._rbm.muV0
            # Define the feature of input.
            self._feature = self._rbm.muH0
            """define the process to generate samples."""
            state = self._rnnCell.zero_state(1, dtype=tf.float32)
            x_ = tf.zeros((1, self._dimInput), dtype='float32')
            # TensorArray to save the output of the generating.
            gen_operator = tf.TensorArray(tf.float32, self.sampleLen)
            # condition and body of while loop (input: i-iteration, xx-RNN input, ss-RNN state)
            i = tf.constant(0)
            cond = lambda i, xx, ss, array: tf.less(i, self.sampleLen)
            #
            def body(i, xx, ss, array):
                ii = i + 1
                hidde_, new_ss = self._rnnCell(self._mlp(xx), ss)
                bvt = tf.tensordot(hidde_, Wdv, [[-1], [0]]) + bv
                bht = tf.tensordot(hidde_, Wdh, [[-1], [0]]) + bh
                new_xx = self._rbm(xx, bvt, bht, k=1)[0]
                new_array = array.write(i, new_xx)
                return ii, new_xx, new_ss, new_array
            gen_operator = tf.while_loop(cond, body, [i, x_, state, gen_operator])[-1]
            self._gen_operator = gen_operator.concat()
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
        logPx_Z = tf.reduce_sum(
            (1 - X) * tf.log(tf.maximum(tf.minimum(1.0, 1 - probs), 1e-32))
            + X * tf.log(tf.maximum(tf.minimum(1.0, probs), 1e-32)),
            axis=[-1])  # shape = [runs, batch, steps]
        logPz = tf.reduce_sum(Pz.log_prob(VAE._Z), axis=[-1])
        return X, logPz_X, logPx_Z, logPz, VAE.x

    """#########################################################################
    ais_function: compute the approximated negative log-likelihood with partition
                  function computed by annealed importance sampling or
                  NVIL with given VAE.
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
                X = np.asarray(X)
                logPz_X = np.asarray(logPz_X)
                logPx_Z = np.asarray(logPx_Z)
                logPz = np.asarray(logPz)
                FEofSample = self._sess.run(self.FEofSample, feed_dict={self.xx: X, self.x: input})
                logTerm = 2 * (-FEofSample + logPz_X - logPx_Z - logPz)
                logTerm_max = np.max(logTerm, axis=0)
                r_ais = np.mean(np.exp(logTerm - logTerm_max), axis=0)
                logZ = 0.5 * (np.log(r_ais+1e-38) + logTerm_max)
                FEofInput = self._sess.run(self.FEofInput, feed_dict={self.x: input})
                loss_value.append(np.mean(FEofInput + logZ))
                loss_value = np.asarray(loss_value).mean()
        return loss_value

"""#########################################################################
Class: gaussRnnRBM - the RNNRBM model for stochastic continuous inputs
                     with Gaussian RBM components.
#########################################################################"""
class gaussRnnRBM(_RnnRBM, object):
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
        with self._graph.as_default():
            # d_t = [batch, steps, hidden]
            self._mlp = MLP(config.init_scale, config.dimIN, config.dimMlp, config.mlpType)
            state = self._rnnCell.zero_state(tf.shape(self.x)[0], dtype=tf.float32)
            d, _ = tf.nn.dynamic_rnn(self._rnnCell, self._mlp(self.x), initial_state=state)
            paddings = tf.constant([[0, 0], [1, 0], [0, 0]])
            dt = tf.pad(d[:, 0:-1, :], paddings)
            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
            with tf.variable_scope("RBM", initializer=initializer):
                bv = tf.get_variable('bv', shape=config.dimIN, initializer=tf.zeros_initializer)
                bh = tf.get_variable('bh', shape=config.dimState, initializer=tf.zeros_initializer)
                Wdv = tf.get_variable('Wdv', shape=[config.dimRec[-1], config.dimIN])
                Wdh = tf.get_variable('Wdh', shape=[config.dimRec[-1], config.dimState])
                bvt = tf.tensordot(dt, Wdv, [[-1], [0]]) + bv
                bht = tf.tensordot(dt, Wdh, [[-1], [0]]) + bh
                # try to learn time variant bias... But fail...
                # Wstd = tf.get_variable('Wstd', shape=[config.dimRec[-1], config.dimInput])
                # bstd = tf.get_variable('bstd', shape=config.dimInput, initializer=tf.zeros_initializer)
                # stdt = tf.tensordot(dt, Wstd, [[-1], [0]]) + bstd
                stdt = 0.5 * tf.ones(shape=config.dimIN)
                self._rbm = gaussRBM(dimV=config.dimIN, dimH=config.dimState, init_scale=config.init_scale,
                                   x=self.x, bv=bvt, bh=bht, std=stdt, k=self._gibbs)
            # the training loss is per frame.
            Loss = self._rbm.ComputeLoss(V=self.x, samplesteps=self._gibbs)
            if VAE is None:
                self._logZ = self._rbm.AIS(self._aisRun, self._aisLevel,
                                           tf.shape(self.x)[0], tf.shape(self.x)[1])
                self._nll = tf.reduce_mean(self._rbm.FreeEnergy(self.x) + self._logZ)
                self.VAE = VAE
            else:
                self._logZ = self._NVIL_VAE(VAE, self._aisRun)  # X, logPz_X, logPx_Z, logPz, VAE.x
                self.xx = tf.placeholder(dtype='float32', shape=[None, None, None, config.dimIN])
                self.FEofSample = self._rbm.FreeEnergy(self.xx)
                self.FEofInput = self._rbm.FreeEnergy(self.x)
                self.VAE = VAE
            self._loss = self._rbm._monitor / self._dimInput        # define the monitor as RMSE/bits.
            self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self._train_step = self._optimizer.minimize(Loss)
            # Define the reconstruction of input.
            self._outputs = self._rbm.muV0
            # Define the feature of input.
            self._feature = self._rbm.muH0
            """define the process to generate samples."""
            state = self._rnnCell.zero_state(1, dtype=tf.float32)
            x_ = tf.zeros((1, self._dimInput), dtype='float32')
            # TensorArray to save the output of the generating.
            gen_operator = tf.TensorArray(tf.float32, self.sampleLen)
            # condition and body of while loop (input: i-iteration, xx-RNN input, ss-RNN state)
            i = tf.constant(0)
            cond = lambda i, xx, ss, array: tf.less(i, self.sampleLen)
            #
            def body(i, xx, ss, array):
                ii = i + 1
                hidde_, new_ss = self._rnnCell(self._mlp(xx), ss)
                bvt = tf.tensordot(hidde_, Wdv, [[-1], [0]]) + bv
                bht = tf.tensordot(hidde_, Wdh, [[-1], [0]]) + bh
                new_xx = self._rbm(xx, bvt=bvt, bht=bht, k=1)[0]
                new_array = array.write(i, new_xx)
                return ii, new_xx, new_ss, new_array

            gen_operator = tf.while_loop(cond, body, [i, x_, state, gen_operator])[-1]
            self._gen_operator = gen_operator.concat()
            self._runSession()


    """#########################################################################
    _NVIL_VAE: generate the graph to compute the NVIL upper bound of log Partition
               function by a well-trained VAE.
    input: VAE - the well-trained VAE(SRNN/VRNN).
           runs - the number of sampling.
    output: the upper boundLogZ.
    #########################################################################"""
    def _NVIL_VAE(self, VAE, runs=100):
        # get the marginal and conditional distribution of the VAE.
        mu, std = VAE._dec
        Px_Z = tf.distributions.Normal(loc=mu, scale=std)
        mu, std = VAE._enc
        Pz_X = tf.distributions.Normal(loc=mu, scale=std)
        mu, std = VAE._prior
        Pz = tf.distributions.Normal(loc=mu, scale=std)
        # generate the samples.
        X = Px_Z.sample(sample_shape=runs)
        logPz_X = tf.reduce_sum(Pz_X.log_prob(VAE._Z), axis=[-1])  # shape = [batch, steps]
        logPx_Z = tf.reduce_sum(Px_Z.log_prob(X), axis=[-1])  # shape = [runs, batch, steps]
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
                X, logPz_X, logPx_Z, logPz = self.VAE._sess.run(self._logZ[0:-1], feed_dict={self._logZ[-1]: input})
                # shape = [runs, batch, steps]
                FEofSample = self._sess.run(self.FEofSample, feed_dict={self.xx: X, self.x: input})
                logTerm = 2 * (-FEofSample + logPz_X - logPx_Z - logPz)
                logTerm_max = np.max(logTerm, axis=0)
                r_ais = np.mean(np.exp(logTerm - logTerm_max), axis=0)
                logZ = 0.5 * (np.log(r_ais) + logTerm_max)
                FEofInput = self._sess.run(self.FEofInput, feed_dict={self.x: input})
                loss_value = np.mean(FEofInput + logZ)
        return loss_value


"""#########################################################################
Class: ssRNNRBM - the RNNRBM model for stochastic continuous inputs
                     with spike-and-slab RBM components.
#########################################################################"""
class ssRNNRBM(_RnnRBM, object):
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
            # d_t = [batch, steps, hidden]
            self._mlp = MLP(config.init_scale, config.dimIN, config.dimMlp, config.mlpType)
            state = self._rnnCell.zero_state(tf.shape(self.x)[0], dtype=tf.float32)
            d, _ = tf.nn.dynamic_rnn(self._rnnCell, self._mlp(self.x), initial_state=state)
            paddings = tf.constant([[0, 0], [1, 0], [0, 0]])
            dt = tf.pad(d[:, 0:-1, :], paddings)
            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
            with tf.variable_scope("ssRBM", initializer=initializer):
                # in ssRNNRBM, the feedback influences only the bias of H.
                bh = tf.get_variable('bh', shape=config.dimState, initializer=tf.zeros_initializer)
                Wdh = tf.get_variable('Wdh', shape=[config.dimRec[-1], config.dimState])
                bht = tf.tensordot(dt, Wdh, [[-1], [0]]) + bh
                bvt = tf.zeros(name='bv', shape=config.dimIN)
                self._rbm = mu_ssRBM(dimV=config.dimIN, dimH=config.dimState,
                                     init_scale=config.init_scale,
                                     x=self.x, bv=bvt, bh=bht, bound=config.Bound,
                                     alphaTrain=config.alphaTrain,
                                     muTrain=config.muTrain,
                                     phiTrain=config.phiTrain,
                                     k=self._gibbs)
            Loss = self._rbm.ComputeLoss(V=self.x, samplesteps=self._gibbs)
            if VAE is None:
                self._logZ = self._rbm.AIS(self._aisRun, self._aisLevel,
                                           tf.shape(self.x)[0], tf.shape(self.x)[1])
                self._nll = tf.reduce_mean(self._rbm.FreeEnergy(self.x) + self._logZ)
                self.VAE = VAE
            else:
                self._logZ = self._NVIL_VAE(VAE, self._aisRun)  # X, logPz_X, logPx_Z, logPz, VAE.x
                self.xx = tf.placeholder(dtype='float32', shape=[None, None, None, config.dimIN])
                self.FEofSample = self._rbm.FreeEnergy(self.xx)
                self.FEofInput = self._rbm.FreeEnergy(self.x)
                self.VAE = VAE
            #
            self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self._train_step = self._optimizer.minimize(Loss)
            # add the computation of precision and covariance matrix of ssRBM.
            self.PreV_h = self._rbm.PreV_h
            self.CovV_h = self._rbm.CovV_h
            # add the tensor computation of reconstruction output.
            self._outputs = self._rbm.muV0
            # add the tensor computation of extracted feature.
            self._feature = self._rbm.muH0
            self._sparse_feature = self._rbm.muH0 * self._rbm.muS0
            # add the monitor
            self._loss = self._rbm._monitor / config.dimIN
            # add the scaling operation of W.
            if config.W_Norm:
                self._scaleW = self._rbm.add_constraint()
            else:
                self._scaleW = None
            """define the process to generate samples."""
            state = self._rnnCell.zero_state(1, dtype=tf.float32)
            x_ = tf.zeros((1, self._dimInput), dtype='float32')
            # TensorArray to save the output of the generating.
            gen_operator = tf.TensorArray(tf.float32, self.sampleLen)
            # condition and body of while loop (input: i-iteration, xx-RNN input, ss-RNN state)
            i = tf.constant(0)
            cond = lambda i, xx, ss, array: tf.less(i, self.sampleLen)

            #
            def body(i, xx, ss, array):
                ii = i + 1
                hidde_, new_ss = self._rnnCell(self._mlp(xx), ss)
                bht = tf.tensordot(hidde_, Wdh, [[-1], [0]]) + bh
                new_xx = self._rbm(xx, bht=bht, k=1)[0]
                new_array = array.write(i, new_xx)
                return ii, new_xx, new_ss, new_array

            gen_operator = tf.while_loop(cond, body, [i, x_, state, gen_operator])[-1]
            self._gen_operator = gen_operator.concat()
            self._runSession()

    """#########################################################################
    convariance: compute the covariance matrix Cv_h.
    input: input - numerical input.
    output:  covariance matrix Cv_h.
    #########################################################################"""
    def convariance(self, input):
        with self._graph.as_default():
            return self._sess.run(self.CovV_h, feed_dict={self.x: input})

    """#########################################################################
    precision: compute the precision matrix Cv_h^{-1}.
    input: input - numerical input.
    output:  precision matrix Cv_h^{-1}.
    #########################################################################"""
    def precision(self, input):
        with self._graph.as_default():
            return self._sess.run(self.PreV_h, feed_dict={self.x: input})

    """#########################################################################
    _NVIL_VAE: generate the graph to compute the NVIL upper bound of log Partition
               function by a well-trained VAE.
    input: VAE - the well-trained VAE(SRNN/VRNN).
           runs - the number of sampling.
    output: the upper boundLogZ.
    #########################################################################"""
    def _NVIL_VAE(self, VAE, runs=100):
        # get the marginal and conditional distribution of the VAE.
        mu, std = VAE._dec
        Px_Z = tf.distributions.Normal(loc=mu, scale=std)
        mu, std = VAE._enc
        Pz_X = tf.distributions.Normal(loc=mu, scale=std)
        mu, std = VAE._prior
        Pz = tf.distributions.Normal(loc=mu, scale=std)
        # generate the samples.
        X = Px_Z.sample(sample_shape=runs)
        logPz_X = tf.reduce_sum(Pz_X.log_prob(VAE._Z), axis=[-1])     # shape = [batch, steps]
        logPx_Z = tf.reduce_sum(Px_Z.log_prob(X), axis=[-1])          # shape = [runs, batch, steps]
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
                X, logPz_X, logPx_Z, logPz = self.VAE._sess.run(self._logZ[0:-1], feed_dict={self._logZ[-1]: input})
                # shape = [runs, batch, steps]
                FEofSample = self._sess.run(self.FEofSample, feed_dict={self.xx: X, self.x: input})
                logTerm = 2 * (-FEofSample + logPz_X - logPx_Z - logPz)
                logTerm_max = np.max(logTerm, axis=0)
                r_ais = np.mean(np.exp(logTerm - logTerm_max), axis=0)
                logZ = 0.5 * (np.log(r_ais) + logTerm_max)
                FEofInput = self._sess.run(self.FEofInput, feed_dict={self.x: input})
                loss_value = np.mean(FEofInput + logZ)
        return loss_value

"""#########################################################################
Class: binssRNNRBM - the RNNRBM model for stochastic binary inputs
                     with spike-and-slab RBM components.
#########################################################################"""
class binssRNNRBM(_RnnRBM, object):
    """#########################################################################
        __init__:the initialization function.
        input: Config - configuration class in ./utility.
               VAE - if a well trained VAE is provided. Using NVIL to estimate the
                     upper bound of the partition function.
        output: None.
        #########################################################################"""

    def __init__(
            self,
            config=configssRNNRBM(),
            VAE=None
    ):
        super().__init__(config)
        """build the graph"""
        with self._graph.as_default():
            # d_t = [batch, steps, hidden]
            self._mlp = MLP(config.init_scale, config.dimIN, config.dimMlp, config.mlpType)
            state = self._rnnCell.zero_state(tf.shape(self.x)[0], dtype=tf.float32)
            d, _ = tf.nn.dynamic_rnn(self._rnnCell, self._mlp(self.x), initial_state=state)
            paddings = tf.constant([[0, 0], [1, 0], [0, 0]])
            dt = tf.pad(d[:, 0:-1, :], paddings)
            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
            with tf.variable_scope("ssRBM", initializer=initializer):
                # in ssRNNRBM, the feedback influences only the bias of H.
                bh = tf.get_variable('bh', shape=config.dimState, initializer=tf.zeros_initializer)
                Wdh = tf.get_variable('Wdh', shape=[config.dimRec[-1], config.dimState])
                bht = tf.tensordot(dt, Wdh, [[-1], [0]]) + bh
                bvt = tf.zeros(name='bv', shape=config.dimIN)
                self._rbm = bin_ssRBM(dimV=config.dimIN, dimH=config.dimState,
                                     init_scale=config.init_scale,
                                     x=self.x, bv=bvt, bh=bht,
                                     alphaTrain=config.alphaTrain,
                                     muTrain=config.muTrain,
                                     k=self._gibbs)
            Loss = self._rbm.ComputeLoss(V=self.x, samplesteps=self._gibbs)
            if VAE is None:
                self._logZ = self._rbm.AIS(self._aisRun, self._aisLevel,
                                           tf.shape(self.x)[0], tf.shape(self.x)[1])
                self._nll = tf.reduce_mean(self._rbm.FreeEnergy(self.x) + self._logZ)
                self.VAE = VAE
            else:
                self._logZ = self._NVIL_VAE(VAE)  # X, logPz_X, logPx_Z, logPz, VAE.x
                self.xx = tf.placeholder(dtype='float32', shape=[None, None, None, config.dimIN])
                self.FEofSample = self._rbm.FreeEnergy(self.xx)
                self.FEofInput = self._rbm.FreeEnergy(self.x)
                self.VAE = VAE
            #
            self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self._train_step = self._optimizer.minimize(Loss)
            # add the tensor computation of reconstruction output.
            self._outputs = self._rbm.muV0
            # add the tensor computation of extracted feature.
            self._feature = self._rbm.muH0
            self._sparse_feature = self._rbm.muH0 * self._rbm.muS0
            # add the monitor
            self._loss = self._rbm._monitor
            # add the scaling operation of W.
            if config.W_Norm:
                self._scaleW = self._rbm.add_constraint()
            else:
                self._scaleW = None
            """define the process to generate samples."""
            state = self._rnnCell.zero_state(1, dtype=tf.float32)
            x_ = tf.zeros((1, self._dimInput), dtype='float32')
            # TensorArray to save the output of the generating.
            gen_operator = tf.TensorArray(tf.float32, self.sampleLen)
            # condition and body of while loop (input: i-iteration, xx-RNN input, ss-RNN state)
            i = tf.constant(0)
            cond = lambda i, xx, ss, array: tf.less(i, self.sampleLen)

            #
            def body(i, xx, ss, array):
                ii = i + 1
                hidde_, new_ss = self._rnnCell(self._mlp(xx), ss)
                bht = tf.tensordot(hidde_, Wdh, [[-1], [0]]) + bh
                new_xx = self._rbm(xx, bht=bht, k=1)[0]
                new_array = array.write(i, new_xx)
                return ii, new_xx, new_ss, new_array

            gen_operator = tf.while_loop(cond, body, [i, x_, state, gen_operator])[-1]
            self._gen_operator = gen_operator.concat()
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
        #logPx_Z = tf.reduce_prod(Px_Z.log_prob(X), axis=[-1])
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
                X = np.asarray(X)
                logPz_X = np.asarray(logPz_X)
                logPx_Z = np.asarray(logPx_Z)
                logPz = np.asarray(logPz)
                FEofSample = self._sess.run(self.FEofSample, feed_dict={self.xx: X, self.x: input})
                logTerm = 2 * (-FEofSample + logPz_X - logPx_Z - logPz)
                logTerm_max = np.max(logTerm, axis=0)
                r_ais = np.mean(np.exp(logTerm - logTerm_max), axis=0)
                logZ = 0.5 * (np.log(r_ais+1e-38) + logTerm_max)
                FEofInput = self._sess.run(self.FEofInput, feed_dict={self.x: input})
                loss_value.append(np.mean(FEofInput + logZ))
                loss_value = np.asarray(loss_value).mean()
        return loss_value

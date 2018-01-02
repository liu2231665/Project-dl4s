"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the file contains the model description of RNN-RBM.
              ----2017.11.03
#########################################################################"""
from dl4s.TRBM import configRNNRBM, configssRNNRBM
from dl4s.SeqVAE.utility import buildRec, MLP
from dl4s.TRBM.RBM import binRBM, gaussRBM, mu_ssRBM, bin_ssRBM
from dl4s.tools import get_batches_idx
import tensorflow as tf
import numpy as np
import time

"""#########################################################################
Class: _RnnRBM - the hyper abstraction of the RnnRBM.
#########################################################################"""
class _RnnRBM(object):
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
        # <tensor graph> define a default graph.
        self._graph = tf.Graph()
        with self._graph.as_default():
            # <tensor placeholder> input.
            self.x = tf.placeholder(dtype='float32', shape=[None, None, config.dimInput])
            # <tensor placeholder> learning rate.
            self.lr = tf.placeholder(dtype='float32', shape=(), name='learningRate')
            # <scalar> the number of samples of AIS.
            self._aisRun = config.aisRun
            # <scalar> the number of intermediate proposal distributions of AIS.
            self._aisLevel = config.aisLevel
            # <scalar> the steps of Gibbs sampling.
            self._gibbs = config.Gibbs
            # <scalar> the size of frame of the input.
            self._dimInput = config.dimInput
            # <scalar> the size of frame of the state.
            self._dimState = config.dimState
            # <string/None> path to save the model.
            self._savePath = config.savePath
            # <string/None> path to save the events.
            self._eventPath = config.eventPath
            # <string/None> path to load the events.
            self._loadPath = config.loadPath
            # <list> collection of trainable parameters.
            self._params = []
            # <list> the RNN components.
            self._rnnCell = buildRec(dimLayer=config.dimRec, unitType=config.recType,
                                     init_scale=config.init_scale)
            #
            self._rbm = None
            self._train_step = None
            self._nll = None
            self._monitor = None
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
        self._sess.run(tf.global_variables_initializer())
        if self._loadPath is not None:
            saver = tf.train.Saver()
            saver.restore(self._sess, self._loadPath)
        return

    """#########################################################################
    train_function: compute the monitor and update the tensor variables.
    input: input - numerical input.
           lrate - <scalar> learning rate.
    output: the reconstruction error.
    #########################################################################"""
    def train_function(self, input, lrate):
        with self._graph.as_default():
            _, loss_value = self._sess.run([self._train_step, self._monitor],
                                           feed_dict={self.x: input, self.lr: lrate})
        return loss_value

    """#########################################################################
    val_function: compute the validation loss with given input.
    input: input - numerical input.
    output: the reconstruction error.
    #########################################################################"""
    def val_function(self, input):
        with self._graph.as_default():
            loss_value = self._sess.run(self._monitor, feed_dict={self.x: input})
        return loss_value

    """#########################################################################
    ais_function: compute the approximated negative log-likelihood with partition
                  function computed by annealed importance sampling.
    input: input - numerical input.
    output: the negative log-likelihood value.
    #########################################################################"""
    def ais_function(self, input):
        with self._graph.as_default():
            loss_value = self._sess.run(self._nll, feed_dict={self.x: input})
        return loss_value

    """#########################################################################
    gen_function: generate samples.
    input: numSteps - the length of the sample sequence.
           gibbs - the number of gibbs sampling.
    output: should be the sample.
    #########################################################################"""
    def gen_function(self, x=None, numSteps=None, gibbs=None):
        newV = x
        with self._graph.as_default():
            if x is not None:
                sample = x
            elif numSteps is not None:
                sample = np.zeros(shape=(1, numSteps, self._dimInput))
            else:
                raise ValueError("Neither input or numSteps is provided!!")
            newV = sample
            k = gibbs if gibbs is not None else self._gibbs
            for i in range(k):
                newV = self._sess.run(self._rbm.newV0, feed_dict={self.x: newV})
        return newV if x is not None else newV[0]

    """#########################################################################
    hidden_function: generate the hidden activation of given X represented by
                     P(H|V).
    input: input - numerical input.
           gibbs - the number of gibbs sampling.
    output: P(H|V).
    #########################################################################"""
    def hidden_function(self, input, gibbs=None):
        newV = input
        with self._graph.as_default():
            k = gibbs if gibbs is not None else self._gibbs
            for i in range(k-1):
                newV = self._sess.run(self._rbm.newV0, feed_dict={self.x: newV})
        return self._sess.run(self._rbm.muH0, feed_dict={self.x: newV})

    """#########################################################################
    output_function: return the conditional mean of the visible layer.
    input: input - numerical input.
    output: P(V|parents).
    #########################################################################"""
    def output_function(self, input):
        with self._graph.as_default():
            return self._sess.run(self._rbm.muV, feed_dict={self.x: input})

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
            config=configRNNRBM(),
            VAE=None
    ):
        super().__init__(config)
        """build the graph"""
        with self._graph.as_default():
            # d_t = [batch, steps, hidden]
            self._mlp = MLP(config.init_scale, config.dimInput, config.dimMlp, config.mlpType)
            state = self._rnnCell.zero_state(tf.shape(self.x)[0], dtype=tf.float32)
            d, _ = tf.nn.dynamic_rnn(self._rnnCell, self._mlp(self.x), initial_state=state)
            paddings = tf.constant([[0, 0], [1, 0], [0, 0]])
            dt = tf.pad(d[:, 0:-1, :], paddings)
            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
            with tf.variable_scope("RBM", initializer=initializer):
                bv = tf.get_variable('bv', shape=config.dimInput, initializer=tf.zeros_initializer)
                bh = tf.get_variable('bh', shape=config.dimState, initializer=tf.zeros_initializer)
                Wdv = tf.get_variable('Wdv', shape=[config.dimRec[-1], config.dimInput])
                Wdh = tf.get_variable('Wdh', shape=[config.dimRec[-1], config.dimState])
                bvt = tf.tensordot(dt, Wdv, [[-1], [0]]) + bv
                bht = tf.tensordot(dt, Wdh, [[-1], [0]]) + bh
            self._rbm = binRBM(dimV=config.dimInput, dimH=config.dimState, init_scale=config.init_scale,
                               x=self.x, bv=bvt, bh=bht, k=self._gibbs)
            # the training loss is per frame.
            self._loss = self._rbm.ComputeLoss(V=self.x, samplesteps=self._gibbs)
            if VAE is None:
                self._logZ = self._rbm.AIS(self._aisRun, self._aisLevel,
                                           tf.shape(self.x)[0], tf.shape(self.x)[1])
                self._nll = tf.reduce_mean(self._rbm.FreeEnergy(self.x) + self._logZ)
                self.VAE = VAE
            else:
                self._logZ = self._NVIL_VAE(VAE)  # X, logPz_X, logPx_Z, logPz, VAE.x
                self.xx = tf.placeholder(dtype='float32', shape=[None, None, None, config.dimInput])
                self.FEofSample = self._rbm.FreeEnergy(self.xx)
                self.FEofInput = self._rbm.FreeEnergy(self.x)
                self.VAE = VAE
            self._monitor = self._rbm._pll
            self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self._train_step = self._optimizer.minimize(self._loss)
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
            config=configRNNRBM(),
            VAE=None
    ):
        super().__init__(config)
        with self._graph.as_default():
            # d_t = [batch, steps, hidden]
            self._mlp = MLP(config.init_scale, config.dimInput, config.dimMlp, config.mlpType)
            state = self._rnnCell.zero_state(tf.shape(self.x)[0], dtype=tf.float32)
            d, _ = tf.nn.dynamic_rnn(self._rnnCell, self._mlp(self.x), initial_state=state)
            paddings = tf.constant([[0, 0], [1, 0], [0, 0]])
            dt = tf.pad(d[:, 0:-1, :], paddings)
            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
            with tf.variable_scope("RBM", initializer=initializer):
                bv = tf.get_variable('bv', shape=config.dimInput, initializer=tf.zeros_initializer)
                bh = tf.get_variable('bh', shape=config.dimState, initializer=tf.zeros_initializer)
                Wdv = tf.get_variable('Wdv', shape=[config.dimRec[-1], config.dimInput])
                Wdh = tf.get_variable('Wdh', shape=[config.dimRec[-1], config.dimState])
                bvt = tf.tensordot(dt, Wdv, [[-1], [0]]) + bv
                bht = tf.tensordot(dt, Wdh, [[-1], [0]]) + bh
                # try to learn time variant bias... But fail...
                # Wstd = tf.get_variable('Wstd', shape=[config.dimRec[-1], config.dimInput])
                # bstd = tf.get_variable('bstd', shape=config.dimInput, initializer=tf.zeros_initializer)
                # stdt = tf.tensordot(dt, Wstd, [[-1], [0]]) + bstd
                stdt = 0.5 * tf.ones(shape=config.dimInput)
                self._rbm = gaussRBM(dimV=config.dimInput, dimH=config.dimState, init_scale=config.init_scale,
                                   x=self.x, bv=bvt, bh=bht, std=stdt, k=self._gibbs)
            # the training loss is per frame.
            self._loss = self._rbm.ComputeLoss(V=self.x, samplesteps=self._gibbs)
            if VAE is None:
                self._logZ = self._rbm.AIS(self._aisRun, self._aisLevel,
                                           tf.shape(self.x)[0], tf.shape(self.x)[1])
                self._nll = tf.reduce_mean(self._rbm.FreeEnergy(self.x) + self._logZ)
                self.VAE = VAE
            else:
                self._logZ = self._NVIL_VAE(VAE, self._aisRun)  # X, logPz_X, logPx_Z, logPz, VAE.x
                self.xx = tf.placeholder(dtype='float32', shape=[None, None, None, config.dimInput])
                self.FEofSample = self._rbm.FreeEnergy(self.xx)
                self.FEofInput = self._rbm.FreeEnergy(self.x)
                self.VAE = VAE
            self._monitor = self._rbm._monitor
            self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self._train_step = self._optimizer.minimize(self._loss)
            self._runSession()

    """#########################################################################
    train_function: compute the monitor and update the tensor variables.
    input: input - numerical input.
           lrate - <scalar> learning rate.
    output: the reconstruction error.
    #########################################################################"""
    def train_function(self, input, lrate):
        with self._graph.as_default():
            _, loss_value = self._sess.run([self._train_step, self._monitor],
                                           feed_dict={self.x: input, self.lr: lrate})
        return loss_value

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
           Bound - the maximum norm of the observation frames.
           VAE - if a well trained VAE is provided. Using NVIL to estimate the
                 upper bound of the partition function.
    output: None.
    #########################################################################"""
    def __init__(
            self,
            config=configssRNNRBM(),
            Bound=(-25.0, 25.0),
            VAE=None
    ):
        super().__init__(config)
        """build the graph"""
        with self._graph.as_default():
            # d_t = [batch, steps, hidden]
            self._mlp = MLP(config.init_scale, config.dimInput, config.dimMlp, config.mlpType)
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
                bvt = tf.zeros(name='bv', shape=config.dimInput)
                self._rbm = mu_ssRBM(dimV=config.dimInput, dimH=config.dimState,
                                     init_scale=config.init_scale,
                                     x=self.x, bv=bvt, bh=bht, bound=Bound,
                                     alphaTrain=config.alphaTrain,
                                     muTrain=config.muTrain,
                                     phiTrain=config.phiTrain,
                                     k=self._gibbs)
            self._loss = self._rbm.ComputeLoss(V=self.x, samplesteps=self._gibbs)
            if VAE is None:
                self._logZ = self._rbm.AIS(self._aisRun, self._aisLevel,
                                           tf.shape(self.x)[0], tf.shape(self.x)[1])
                self._nll = tf.reduce_mean(self._rbm.FreeEnergy(self.x) + self._logZ)
                self.VAE = VAE
            else:
                self._logZ = self._NVIL_VAE(VAE, self._aisRun)  # X, logPz_X, logPx_Z, logPz, VAE.x
                self.xx = tf.placeholder(dtype='float32', shape=[None, None, None, config.dimInput])
                self.FEofSample = self._rbm.FreeEnergy(self.xx)
                self.FEofInput = self._rbm.FreeEnergy(self.x)
                self.VAE = VAE
            #
            self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self._train_step = self._optimizer.minimize(self._loss)
            # add the computation of precision and covariance matrix of ssRBM.
            self.PreV_h = self._rbm.PreV_h
            self.CovV_h = self._rbm.CovV_h
            # add the monitor
            self._monitor = self._rbm._monitor
            # add the scaling operation of W.
            if config.W_Norm:
                self._scaleW = self._rbm.add_constraint()
            else:
                self._scaleW = None
            self._runSession()

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

    """#########################################################################
    sparse_hidden_function: generate the sparse hidden activation of given X
                      represented by E(H*S|V).
    input: input - numerical input.
           gibbs - the number of gibbs sampling.
    output: E(H*S|V).
    #########################################################################"""
    def sparse_hidden_function(self, input, gibbs=None):
        newV = input
        with self._graph.as_default():
            k = gibbs if gibbs is not None else self._gibbs
            for i in range(k - 1):
                newV = self._sess.run(self._rbm.newV0, feed_dict={self.x: newV})
        meanH, meanS = self._sess.run([self._rbm.muH0, self._rbm.muS0], feed_dict={self.x: newV})
        return meanH * meanS



    """#########################################################################
    train_function: compute the monitor and update the tensor variables.
    input: input - numerical input.
           lrate - <scalar> learning rate.
    output: the reconstruction error.
    #########################################################################"""
    def train_function(self, input, lrate):
        with self._graph.as_default():
            _, loss_value = self._sess.run([self._train_step, self._monitor],
                                           feed_dict={self.x: input, self.lr: lrate})
            if self._scaleW is not None:
                newW = self._sess.run(self._scaleW)
        return loss_value

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
            self._mlp = MLP(config.init_scale, config.dimInput, config.dimMlp, config.mlpType)
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
                bvt = tf.zeros(name='bv', shape=config.dimInput)
                self._rbm = bin_ssRBM(dimV=config.dimInput, dimH=config.dimState,
                                     init_scale=config.init_scale,
                                     x=self.x, bv=bvt, bh=bht,
                                     alphaTrain=config.alphaTrain,
                                     muTrain=config.muTrain,
                                     k=self._gibbs)
            self._loss = self._rbm.ComputeLoss(V=self.x, samplesteps=self._gibbs)
            if VAE is None:
                self._logZ = self._rbm.AIS(self._aisRun, self._aisLevel,
                                           tf.shape(self.x)[0], tf.shape(self.x)[1])
                self._nll = tf.reduce_mean(self._rbm.FreeEnergy(self.x) + self._logZ)
                self.VAE = VAE
            else:
                self._logZ = self._NVIL_VAE(VAE)  # X, logPz_X, logPx_Z, logPz, VAE.x
                self.xx = tf.placeholder(dtype='float32', shape=[None, None, None, config.dimInput])
                self.FEofSample = self._rbm.FreeEnergy(self.xx)
                self.FEofInput = self._rbm.FreeEnergy(self.x)
                self.VAE = VAE
                pass
            #
            self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self._train_step = self._optimizer.minimize(self._loss)
            # add the monitor
            self._monitor = self._rbm._monitor
            # add the scaling operation of W.
            if config.W_Norm:
                self._scaleW = self._rbm.add_constraint()
            else:
                self._scaleW = None
            self._runSession()

    """#########################################################################
    sparse_hidden_function: generate the sparse hidden activation of given X
                      represented by E(H*S|V).
    input: input - numerical input.
           gibbs - the number of gibbs sampling.
    output: E(H*S|V).
    #########################################################################"""
    def sparse_hidden_function(self, input, gibbs=None):
        newV = input
        with self._graph.as_default():
            k = gibbs if gibbs is not None else self._gibbs
            for i in range(k - 1):
                newV = self._sess.run(self._rbm.newV0, feed_dict={self.x: newV})
        meanH, meanS = self._sess.run([self._rbm.muH0, self._rbm.muS0], feed_dict={self.x: newV})
        return meanH * meanS

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

    """#########################################################################
    val_function: compute the validation loss with given input.
    input: input - numerical input.
    output: the bound of -log P(x).
    #########################################################################"""
    def val_function(self, input):
        if self.VAE is not None:
            return self.ais_function(input)
        else:
            with self._graph.as_default():
                loss_value = self._sess.run(self._monitor, feed_dict={self.x: input})
            return loss_value

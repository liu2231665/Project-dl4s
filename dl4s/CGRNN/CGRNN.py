"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the file contains the model description of CGRNN.
              ----2017.11.15
#########################################################################"""
from .utility import configCGRNN, CGCell
from dl4s.tools import BernoulliNLL
import tensorflow as tf
import numpy as np

"""#########################################################################
Class: _CGRNN - the hyper abstraction of the CGRNN.
#########################################################################"""
class _CGRNN(object):
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

        # <tensor graph> define a default graph.
        self._graph = tf.Graph()
        with self._graph.as_default():
            # <scalar> the steps of Gibbs sampling.
            self._gibbs = config.Gibbs
            # <tensor placeholder> input.
            self.x = tf.placeholder(dtype='float32', shape=[None, None, config.dimInput])
            # <tensor placeholder> learning rate.
            self.lr = tf.placeholder(dtype='float32', shape=(), name='learningRate')
            # <scalar> the number of samples of AIS.
            self._aisRun = config.aisRun
            # <scalar> the number of intermediate proposal distributions of AIS.
            self._aisLevel = config.aisLevel
            # <scalar> dimensions of input frame.
            self._dimInput = config.dimInput
            # <scalar> dimensions of stochastic states.
            self._dimState = config.dimState
            # <scalar list> the size of forward recurrent hidden layers.
            self._dimRec = config.dimRec
            # <scalar list> the size of feed-forward hidden layers.
            self._dimMlp = config.dimMlp
            # <string> the mode.
            self._mode = config.mode
            # <scalar list> the size of backward recurrent hidden layers/MLP depended on the mode.
            self._savePath = config.savePath
            # <string/None> path to save the events.
            self._eventPath = config.eventPath
            # <string/None> path to load the events.
            self._loadPath = config.loadPath
            # <list> collection of trainable parameters.
            self._params = []
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
Class: binCGRNN - the CGRNN mode for binary input.
#########################################################################"""
class binCGRNN(_CGRNN, object):
    """#########################################################################
    __init__:the initialization function.
    input: Config - configuration class in ./utility.
           binRNN - if a well trained binRNN is provided. Using NVIL to estimate the
                 upper bound of the partition function.
    output: None.
    #########################################################################"""
    def __init__(
            self,
            config=configCGRNN(),
            binRNN=None
    ):
        super().__init__(config)
        """build the graph"""
        with self._graph.as_default():
            self.Cell = CGCell(config, inputType='binary')
            state = self.Cell.zero_state(tf.shape(self.x)[0], dtype=tf.float32)
            (self.newV, self.newH, self.newS, self.muV, self.muH, self.muS,
             self.bvt, self.bht), _ = tf.nn.dynamic_rnn(self.Cell, self.x, initial_state=state)
            # update the RBM's bias with bvt & bht.
            self.Cell.RBM._bh = self.bht
            self.Cell.RBM._bv = self.bvt
            # the training loss is per frame.
            self._loss = self.Cell.RBM.ComputeLoss(V=self.x, samplesteps=config.Gibbs)
            self._monitor = BernoulliNLL(self.x, self.muV) * config.dimInput
            self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self._train_step = self._optimizer.minimize(self._loss)
            #
            if binRNN is None:
                self._logZ = self.Cell.RBM.AIS(self._aisRun, self._aisLevel,
                                           tf.shape(self.x)[0], tf.shape(self.x)[1])
                self._nll = tf.reduce_mean(self.Cell.RBM.FreeEnergy(self.x) + self._logZ)
                self.binRNN = binRNN
            else:
                self._logZ = self._NVIL_RNN(binRNN, config.aisRun)  # X, logPz_X, logPx_Z, logPz, VAE.x
                self.xx = tf.placeholder(dtype='float32', shape=[None, None, None, config.dimInput])
                self.FEofSample = self.Cell.RBM.FreeEnergy(self.xx)
                self.FEofInput = self.Cell.RBM.FreeEnergy(self.x)
                self.binRNN = binRNN
            #
            self._runSession()
            pass

    """#########################################################################
    hidden_function: generate the hidden activation of given X represented by
                     P(H|V).
    input: input - numerical input.
           gibbs - the number of gibbs sampling.
    output: P(H|V).
    #########################################################################"""
    def hidden_function(self, input, gibbs=None):
        # update the RBM's bias with bvt & bht.
        self.Cell.RBM._bh = self.bht
        self.Cell.RBM._bv = self.bvt
        #
        newV = input
        with self._graph.as_default():
            k = gibbs if gibbs is not None else self._gibbs
            if k == self._gibbs:
                return self._sess.run(self.muH, feed_dict={self.x: newV})
            else:
                for i in range(k - 1):
                    newV = self._sess.run(self.Cell.RBM.newV0, feed_dict={self.x: newV})
                return self._sess.run(self.Cell.RBM.muH0, feed_dict={self.x: newV})

    """#########################################################################
    gen_function: generate samples.
    input: numSteps - the length of the sample sequence.
           gibbs - the number of gibbs sampling.
    output: should be the sample.
    #########################################################################"""
    def gen_function(self, x=None, numSteps=None, gibbs=None):
        # update the RBM's bias with bvt & bht.
        self.Cell.RBM._bh = self.bht
        self.Cell.RBM._bv = self.bvt
        #
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
            if k == self._gibbs:
                newV = self._sess.run(self.newV, feed_dict={self.x: newV})
            else:
                for i in range(k):
                    newV = self._sess.run(self.Cell.RBM.newV0, feed_dict={self.x: newV})
        return newV if x is not None else newV[0]

    """#########################################################################
    _NVIL_VAE: generate the graph to compute the NVIL upper bound of log Partition
               function by a well-trained VAE.
    input: RNN - the well-trained binary autoregressive RNN.
           runs - the number of samples.
    output: the upper boundLogZ.
    #########################################################################"""
    def _NVIL_RNN(self, binRNN, runs=100):
        # get the marginal and conditional distribution of the VAE.
        probs = binRNN._outputs
        Px_Z = tf.distributions.Bernoulli(probs=probs, dtype=tf.float32)
        # generate the samples.
        X = Px_Z.sample(runs)
        logPx = tf.reduce_sum(
            (1 - X) * tf.log(tf.maximum(tf.minimum(1.0, 1 - probs), 1e-32))
            + X * tf.log(tf.maximum(tf.minimum(1.0, probs), 1e-32)),
            axis=[-1])  # shape = [runs, batch, steps]
        return X, logPx, binRNN.x

    """#########################################################################
    ais_function: compute the approximated negative log-likelihood with partition
                  function computed by annealed importance sampling.
    input: input - numerical input.
    output: the negative log-likelihood value.
    #########################################################################"""
    def ais_function(self, input):
        with self._graph.as_default():
            if self.binRNN is None:
                loss_value = self._sess.run(self._nll, feed_dict={self.x: input})
            else:
                loss_value = []
                X, logPx = self.binRNN._sess.run(self._logZ[0:-1], feed_dict={self._logZ[-1]: input})
                FEofSample = self._sess.run(self.FEofSample, feed_dict={self.xx: X, self.x: input})
                logTerm = 2 * (-FEofSample - logPx)
                logTerm_max = np.max(logTerm, axis=0)
                r_ais = np.mean(np.exp(logTerm - logTerm_max), axis=0)
                logZ = 0.5 * (np.log(r_ais+1e-38) + logTerm_max)
                FEofInput = self._sess.run(self.FEofInput, feed_dict={self.x: input})
                loss_value.append(np.mean(FEofInput + logZ))
                loss_value = np.asarray(loss_value).mean()
        return loss_value
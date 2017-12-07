"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the file contains the model description of RNN-RBM.
              ----2017.11.03
#########################################################################"""
from dl4s.TRBM import configRNNRBM, configssRNNRBM
from dl4s.SeqVAE.utility import buildRec, MLP
from dl4s.TRBM.RBM import binRBM, gaussRBM, mu_ssRBM
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
    saveModel:save the trained model into disk.
    input: savePath - another saving path, if not provide, use the default path.
    output: None
    #########################################################################"""
    def saveModel(self, savePath=None):
        with self._graph.as_default():
            # Create a saver.
            saver = tf.train.Saver()
            if savePath is None:
                saver.save(self._sess, self._savePath)
            else:
                saver.save(self._sess, savePath)
        return

    """#########################################################################
    loadModel:load the model from disk.
    input: loadPath - another loading path, if not provide, use the default path.
    output: None
    #########################################################################"""
    def loadModel(self, loadPath=None):
        with self._graph.as_default():
            # Create a saver.
            saver = tf.train.Saver()
            if loadPath is None:
                if self._loadPath is not None:
                    saver.restore(self._sess, self._loadPath)
                else:
                    raise (ValueError("No loadPath is given!"))
            else:
                saver.restore(self._sess, loadPath)
        return

    """#########################################################################
    saveEvent:save the event to visualize the last model once.
              (To visualize other aspects, other codes should be used.)
    input: None
    output: None
    #########################################################################"""
    def saveEvent(self):
        if self._eventPath is None:
            raise ValueError("Please privide the path to save the events by self._eventPath!!")
        with self._graph.as_default():
            # compute the statistics of the parameters.
            for param in self._params:
                scopeName = param.name.split('/')[-1]
                with tf.variable_scope(scopeName[0:-2]):
                    mean = tf.reduce_mean(param)
                    tf.summary.scalar('mean', mean)
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(param - mean)))
                    tf.summary.scalar('stddev', stddev)
                    tf.summary.scalar('max', tf.reduce_max(param))
                    tf.summary.scalar('min', tf.reduce_min(param))
                    tf.summary.histogram('histogram', param)
            # visualize the trainable parameters of the model.
            summary = tf.summary.merge_all()
            summary_str = self._sess.run(summary)
            # Define a event writer and write the events into disk.
            Writer = tf.summary.FileWriter(self._eventPath, self._sess.graph)
            Writer.add_summary(summary_str)
            Writer.flush()
            Writer.close()
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
        if x is not None:
            sample = x
        elif numSteps is not None:
            sample = np.zeros(shape=(1, numSteps, self._dimInput))
        else:
            raise ValueError("Neither input or numSteps is provided!!")
        k = gibbs if gibbs is not None else self._gibbs
        newV= self._rbm.GibbsSampling(self.x, k=k)[0]
        sample = self._sess.run(newV, feed_dict={self.x: sample})
        return sample if x is not None else sample[0]

    """#########################################################################
    hidden_function: generate the hidden activation of given X represented by
                     P(H|V).
    input: input - numerical input.
           gibbs - the number of gibbs sampling.
    output: P(H|V).
    #########################################################################"""
    def hidden_function(self, input, gibbs=None):
        k = gibbs if gibbs is not None else self._gibbs
        _, _, Ph_v, _ = self._rbm.GibbsSampling(self.x, k=k)
        return self._sess.run(Ph_v, feed_dict={self.x: input})

    """#########################################################################
    full_train: define to fully train a model given the dataset.
    input: dataset - the dataset used to train. The split should be train/valid
                     /test.
           maxEpoch - the maximum epoches to train the model.
           batchSize - the batch size for training.
           earlyStop - the tolerance epoches for early stopping.
           learning_rate - you know what it is...
           saveto - the additional save path that may be different from
                    the default to save the history loss during training.
           valid_batchSize - the batch size for validation and testing.
    output: None.
    #########################################################################"""
    def full_train(self, dataset, maxEpoch, batchSize, earlyStop,
                   learning_rate, saveto, valid_batchSize=1):
        # slit the dataset.
        trainData = dataset['train']
        validData = dataset['valid']
        testData = dataset['test']
        validBatch = get_batches_idx(len(validData), valid_batchSize, False)
        testBatch = get_batches_idx(len(testData), valid_batchSize, False)

        historyLoss = []  # <list> record the training process.
        durations = []  # <list> record the training duration.
        worseCase = 0  # indicate the worse cases for early stopping
        bestEpoch = -1

        # Issue: the idx should be a list. Hence, .tolist() is required.
        for epoch in range(maxEpoch):
            start_time = time.time()  # the start time of epoch.
            # update the model w.r.t the training set and record the average loss.
            trainLoss = []
            trainBatch = get_batches_idx(len(trainData), batchSize, True)
            for Idx in trainBatch:
                x = trainData[Idx.tolist()]
                trainLoss.append(x.shape[0] * self.train_function(x, learning_rate))
            trainLoss_avg = np.asarray(trainLoss).sum() / len(trainData)

            duration = time.time() - start_time  # the duration of one epoch.
            durations.append(duration)

            # evaluate the model w.r.t the valid set and record the average loss.
            validLoss = []
            for Idx in validBatch:
                x = validData[Idx.tolist()]
                validLoss.append(x.shape[0] * self.val_function(x))
            validLoss_avg = np.asarray(validLoss).sum() / len(validData)
            print("In epoch \x1b[1;32m%4d\x1b[0m: the training loss is "
                  "\x1b[1;32m%10.4f\x1b[0m; the valid loss is \x1b[1;32m%10.4f\x1b[0m." % (
                      epoch, trainLoss_avg, validLoss_avg))

            # check the early stopping conditions.
            if len(historyLoss) == 0 or validLoss_avg < np.min(np.asarray(historyLoss)[:, 1]):
                worseCase = 0
                bestEpoch = epoch
                self.saveModel()
            else:
                worseCase += 1
            historyLoss.append([trainLoss_avg, validLoss_avg])
            if worseCase >= earlyStop:
                break

        durationEpoch = np.asarray(durations).mean()
        print('The average epoch duration is \x1b[1;91m%10.4f\x1b[0m seconds.' % durationEpoch)

        # evaluate the best model w.r.t the test set and record the average loss.
        self.loadModel(self._savePath)

        trainLoss = []
        trainBatch = get_batches_idx(len(trainData), batchSize, True)
        for Idx in trainBatch:
            x = trainData[Idx.tolist()]
            trainLoss.append(x.shape[0] * self.val_function(x))
        trainLoss_avg = np.asarray(trainLoss).sum() / len(trainData)

        validLoss = []
        for Idx in validBatch:
            x = validData[Idx.tolist()]
            validLoss.append(x.shape[0] * self.val_function(x))
        validLoss_avg = np.asarray(validLoss).sum() / len(validData)

        testLoss = []
        for Idx in testBatch:
            x = testData[Idx.tolist()]
            testLoss.append(x.shape[0] * self.val_function(x))
        testLoss_avg = np.asarray(testLoss).sum() / len(testData)

        # evaluate the model w.r.t the valid set and record the average loss.
        print("BEST MODEL from epoch \x1b[1;91m%4d\x1b[0m with training loss"
              " \x1b[1;91m%10.4f\x1b[0m and valid loss \x1b[1;91m%10.4f\x1b[0m."
              % (bestEpoch, trainLoss_avg, validLoss_avg))
        print('The testing loss is \x1b[1;91m%10.4f\x1b[0m.' % testLoss_avg)

        if saveto is not None:
            np.savez(saveto, historyLoss=np.asarray(historyLoss),
                     testLoss=testLoss_avg, durationEpoch=durationEpoch)
        return

"""#########################################################################
Class: binRnnRBM - the RNNRBM model for stochastic binary inputs.
#########################################################################"""
class binRnnRBM(_RnnRBM, object):
    """#########################################################################
    __init__:the initialization function.
    input: Config - configuration class in ./utility.
    output: None.
    #########################################################################"""
    def __init__(
            self,
            config=configRNNRBM()
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
            self._monitor = self._rbm._pll
            self._logZ = self._rbm.AIS(self._aisRun, self._aisLevel, tf.shape(self.x)[0], tf.shape(self.x)[1])
            self._nll = tf.reduce_mean(self._rbm.FreeEnergy(self.x) + self._logZ)
            self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self._train_step = self._optimizer.minimize(self._loss)
            self._runSession()


"""#########################################################################
Class: gaussRnnRBM - the RNNRBM model for stochastic continuous inputs
                     with Gaussian RBM components.
#########################################################################"""
class gaussRnnRBM(_RnnRBM, object):
    """#########################################################################
    __init__:the initialization function.
    input: Config - configuration class in ./utility.
    output: None.
    #########################################################################"""
    def __init__(
            self,
            config=configRNNRBM()
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
            self._monitor = self._rbm._monitor
            self._logZ = self._rbm.AIS(self._aisRun, self._aisLevel, tf.shape(self.x)[0], tf.shape(self.x)[1])
            self._nll = tf.reduce_mean(self._rbm.FreeEnergy(self.x) + self._logZ)
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
Class: ssRNNRBM - the RNNRBM model for stochastic continuous inputs
                     with spike-and-slab RBM components.
#########################################################################"""
class ssRNNRBM(_RnnRBM, object):
    """#########################################################################
    __init__:the initialization function.
    input: Config - configuration class in ./utility.
    output: None.
    #########################################################################"""
    def __init__(
            self,
            config=configssRNNRBM()
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
                                     x=self.x, bv=bvt, bh=bht,
                                     alphaTrain=config.alphaTrain,
                                     muTrain=config.muTrain,
                                     phiTrain=config.phiTrain,
                                     k=self._gibbs)
            self._loss = self._rbm.ComputeLoss(V=self.x, samplesteps=self._gibbs)
            self._logZ = self._rbm.AIS(self._aisRun, self._aisLevel, tf.shape(self.x)[0], tf.shape(self.x)[1])
            self._nll = tf.reduce_mean(self._rbm.FreeEnergy(self.x) + self._logZ)
            self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self._train_step = self._optimizer.minimize(self._loss)
            # add the computation of precision and covariance matrix of ssRBM.
            self.PreV_h = self._rbm.PreV_h
            self.CovV_h = self._rbm.CovV_h
            # add the scaling operation of W.
            if config.W_Norm:
                self._scaleW = self._rbm.add_constraint()
            else:
                self._scaleW = None
            self._runSession()

    """#########################################################################
    train_function: compute the monitor and update the tensor variables.
    input: input - numerical input.
           lrate - <scalar> learning rate.
    output: the reconstruction error.
    #########################################################################"""
    def train_function(self, input, lrate):
        with self._graph.as_default():
            self._sess.run(self._train_step, feed_dict={self.x: input, self.lr: lrate})
            if self._scaleW is not None:
                newW = self._sess.run(self._scaleW)
            sample = self.gen_function(input)
            rmse = (input - sample) ** 2
            rmse = rmse.sum(-1)
            return np.sqrt(rmse.mean())

    """#########################################################################
    val_function: compute the validation loss with given input.
    input: input - numerical input.
    output: the reconstruction error.
    #########################################################################"""
    def val_function(self, input):
        with self._graph.as_default():
            sample = self.gen_function(input)
            rmse = (input - sample) ** 2
            rmse = rmse.sum(-1)
            return np.sqrt(rmse.mean())

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
    hidden_function: generate the hidden activation of given X represented by
                     P(H|V).
    input: input - numerical input.
           gibbs - the number of gibbs sampling.
    output: P(H|V).
    #########################################################################"""
    def hidden_function(self, input, gibbs=None):
        k = gibbs if gibbs is not None else self._gibbs
        Ph_v = self._rbm.GibbsSampling(self.x, k=k)[4]
        return self._sess.run(Ph_v, feed_dict={self.x: input})


"""#########################################################################
Class: binssRNNRBM - the RNNRBM model for stochastic binary inputs
                     with spike-and-slab RBM components.
#########################################################################"""
class binssRNNRBM(_RnnRBM, object):
    pass
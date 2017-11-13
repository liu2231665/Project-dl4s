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
    recognitionOutput: compute the P(Z|X) with given X.
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
            print("In epoch \x1b[1;32m%4d\x1b[0m: the training ELBO is "
                  "\x1b[1;32m%10.4f\x1b[0m; the valid ELBO is \x1b[1;32m%10.4f\x1b[0m." % (
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
        print("BEST MODEL from epoch \x1b[1;91m%4d\x1b[0m with training ELBO"
              " \x1b[1;91m%10.4f\x1b[0m and valid ELBO \x1b[1;91m%10.4f\x1b[0m."
              % (bestEpoch, trainLoss_avg, validLoss_avg))
        print('The testing ELBO is \x1b[1;91m%10.4f\x1b[0m.' % testLoss_avg)

        if saveto is not None:
            np.savez(saveto, historyLoss=np.asarray(historyLoss),
                     testLoss=testLoss_avg, durationEpoch=durationEpoch)
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
                    x_ = tf.distributions.Normal(loc=mu, scale=tf.sqrt(sig)).sample()
                    samples.append(x_)
            samples = tf.concat(samples, 0)
        return self._sess.run(samples)
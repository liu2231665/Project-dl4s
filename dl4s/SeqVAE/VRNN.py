# TODO:
"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the file contains the model description of VRNN.
              ----2017.11.13
#########################################################################"""

import tensorflow as tf
from .utility import buildVRNN
from .utility import configVRNN
from . import GaussKL, BernoulliNLL, GaussNLL
from dl4s.tools import get_batches_idx
import numpy as np
import time, os

"""#########################################################################
Class: _VRNN - the hyper abstraction of the VRNN.
#########################################################################"""
class _VRNN(object):
    """#########################################################################
    __init__:the initialization function.
    input: Config - configuration class in ./utility.
    output: None.
    #########################################################################"""
    def __init__(
            self,
            config=configVRNN()
    ):
        # Check the dimension configuration.
        if config.dimRec == []:
            raise (ValueError('The recurrent structure is empty!'))
        # <tensor graph> define a default graph.
        self._graph = tf.Graph()
        with self._graph.as_default():
            # <tensor placeholder> input.
            self.x = tf.placeholder(dtype='float32', shape=[None, None, config.dimInput])
            # <tensor placeholder> learning rate.
            self.lr = tf.placeholder(dtype='float32', shape=(), name='learningRate')
            # <scalar list> the size of recurrent hidden layers.
            self._dimRec = config.dimRec
            # <scalar list> the size of feedforward hidden layers of input.
            self._dimForX = config.dimForX
            # <scalar list> the size of feedforward hidden layers of stochastic layer.
            self._dimForZ = config.dimForZ
            # <scalar> dimensions of input frame.
            self._dimInput = config.dimInput
            # <scalar> dimensions of stochastic states.
            self._dimState = config.dimState
            # <string/None> path to save the model.
            self._savePath = config.savePath
            # <string/None> path to save the events.
            self._eventPath = config.eventPath
            # <string/None> path to load the events.
            self._loadPath = config.loadPath
            # <list> collection of trainable parameters.
            self._params = []

            self._prior_mu, self._prior_sig, self._pos_mu, \
            self._pos_sig, self._hidden_dec, self._h_tm1, self._varCell = buildVRNN(self.x, self._graph, config)
            self._loss = GaussKL(self._pos_mu, self._pos_sig ** 2, self._prior_mu, self._prior_sig ** 2)
            self._kl_divergence = self._loss
            # <pass> will be define in the children classes.
            self._train_step = None
            # the prior P(Z).
            self._prior = [self._prior_mu, self._prior_sig]
            # the posterior P(Z|X).
            self._enc = [self._pos_mu, self._pos_sig]
            # <pass> compute the posterior P(X|Z)
            self._dec = []

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
            _, loss_value = self._sess.run([self._train_step, self._loss],
                                           feed_dict={self.x: input, self.lr: lrate})
        return loss_value * input.shape[-1]

    """#########################################################################
    val_function: compute the loss with given input.
    input: input - numerical input.
    output: the loss value.
    #########################################################################"""
    def val_function(self, input):
        with self._graph.as_default():
            loss_value = self._sess.run(self._loss, feed_dict={self.x: input})
        return loss_value * input.shape[-1]

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
    gen_function: generate samples.
    input: numSteps - the length of the sample sequence.
    output: should be the sample.
    #########################################################################"""
    def gen_function(self, numSteps):
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
Class: binSTORN - the VRNN model for stochastic binary inputs.
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
                Wdec = tf.get_variable('Wdec', shape=(self._hidden_dec.shape[-1], config.dimInput))
                bdec = tf.get_variable('bdec', shape=config.dimInput, initializer=tf.zeros_initializer)
                self._dec = tf.nn.sigmoid(tf.tensordot(self._hidden_dec, Wdec, [[-1], [0]]) + bdec)
                self._loss += BernoulliNLL(self.x, self._dec)
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
            # Set the variational cell to use the prior P(Z) to generate Zt.
            self._varCell.setGen()
            #
            state = self._varCell.zero_state(1, dtype=tf.float32)
            x_ = tf.zeros((1, self._dimInput), dtype='float32')
            samples = []
            with tf.variable_scope('logit', reuse=True):
                Wdec = tf.get_variable('Wdec')
                bdec = tf.get_variable('bdec')

            for i in range(numSteps):
                (_, _, _, _, hidde_, _), state = self._varCell(x_, state)
                probs = tf.nn.sigmoid(tf.nn.xw_plus_b(hidde_, Wdec, bdec))
                x_ = tf.distributions.Bernoulli(probs=probs, dtype=tf.float32).sample()
                samples.append(x_)
            samples = tf.concat(samples, 0)
        return self._sess.run(samples)

    """#########################################################################
    output_function: reconstruction function.
    input: input - .
    output: should be the reconstruction represented by the probability.
    #########################################################################"""
    def output_function(self, input):
            return self._sess.run(self._dec, feed_dict={self.x: input})

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
                Wdec_mu = tf.get_variable('Wdec_mu', shape=(self._hidden_dec.shape[-1], config.dimInput))
                bdec_mu = tf.get_variable('bdec_mu', shape=config.dimInput, initializer=tf.zeros_initializer)
                mu = tf.tensordot(self._hidden_dec, Wdec_mu, [[-1], [0]]) + bdec_mu
                Wdec_sig = tf.get_variable('Wdec_sig', shape=(self._hidden_dec.shape[-1], config.dimInput))
                bdec_sig = tf.get_variable('bdec_sig', shape=config.dimInput, initializer=tf.zeros_initializer)
                std = tf.nn.softplus(tf.tensordot(self._hidden_dec, Wdec_sig, [[-1], [0]]) + bdec_sig) + 1e-8
                self._dec = [mu, std]

                self._loss += GaussNLL(self.x, mu, std**2)
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
            # Set the variational cell to use the prior P(Z) to generate Zt.
            self._varCell.setGen()
            #
            state = self._varCell.zero_state(1, dtype=tf.float32)
            x_ = tf.zeros((1, self._dimInput), dtype='float32')
            samples = []
            with tf.variable_scope('output', reuse=True):
                # compute the mean and standard deviation of P(X|Z).
                Wdec_mu = tf.get_variable('Wdec_mu')
                bdec_mu = tf.get_variable('bdec_mu')
                Wdec_sig = tf.get_variable('Wdec_sig')
                bdec_sig = tf.get_variable('bdec_sig')

            for i in range(numSteps):
                (_, _, _, _, hidde_, _), state = self._varCell(x_, state)
                mu = tf.tensordot(hidde_, Wdec_mu, [[-1], [0]]) + bdec_mu
                std = tf.nn.softplus(tf.tensordot(hidde_, Wdec_sig, [[-1], [0]]) + bdec_sig) + 1e-8
                x_ = tf.distributions.Normal(loc=mu, scale=std).sample()
                samples.append(x_)
            samples = tf.concat(samples, 0)
        return self._sess.run(samples)

    """#########################################################################
    output_function: reconstruction function.
    input: input - .
    output: should be the reconstruction represented by the probability.
    #########################################################################"""
    def output_function(self, input):
        mean, std = self._sess.run(self._dec, feed_dict={self.x: input})
        return mean, std
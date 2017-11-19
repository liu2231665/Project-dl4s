# TODO:
"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the file contains the model description of SRNN.
              ----2017.11.15
#########################################################################"""
from dl4s.tools import GaussKL, BernoulliNLL
from dl4s.SeqVAE import configSRNN
from dl4s.SeqVAE.utility import buildSRNN
import tensorflow as tf


"""#########################################################################
Class: _SRNN - the hyper abstraction of the SRNN.
#########################################################################"""
class _SRNN(object):
    """#########################################################################
        __init__:the initialization function.
        input: Config - configuration class in ./utility.
        output: None.
        #########################################################################"""

    def __init__(
            self,
            config=configSRNN()
    ):
        # Check the froward recurrent dimension configuration.
        if config.dimRecD == []:
            raise (ValueError('The forward recurrent structure is empty!'))
        # Check the froward recurrent dimension configuration.
        if config.dimRecA == [] and config.mode == 'smooth':
            raise (ValueError('The backward recurrent structure in smooth mode is empty!'))

        # <tensor graph> define a default graph.
        self._graph = tf.Graph()
        with self._graph.as_default():
            # <tensor placeholder> input.
            self.x = tf.placeholder(dtype='float32', shape=[None, None, config.dimInput])
            # <tensor placeholder> learning rate.
            self.lr = tf.placeholder(dtype='float32', shape=(), name='learningRate')
            # <scalar> dimensions of input frame.
            self._dimInput = config.dimInput
            # <scalar> dimensions of stochastic states.
            self._dimState = config.dimState
            # <scalar list> the size of forward recurrent hidden layers.
            self._dimRecD = config.dimRecD
            # <scalar list> the size of backward recurrent hidden layers/MLP depended on the mode.
            self._dimRecA = config.dimRecA
            # <string/None> path to save the model.
            self._savePath = config.savePath
            # <string/None> path to save the events.
            self._eventPath = config.eventPath
            # <string/None> path to load the events.
            self._loadPath = config.loadPath
            # <list> collection of trainable parameters.
            self._params = []
            # components of the SRNN.
            self._prior_mu, self._prior_sig, self._pos_mu, self._pos_sig, self._hidden_dec = \
                buildSRNN(self.x, self._graph, config)
            # the loss functions.
            self._loss = GaussKL(self._pos_mu, self._pos_sig ** 2, self._prior_mu, self._prior_sig ** 2)
            self._kl_divergence = self._loss
            #
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
Class: _SRNN - the SRNN model for stochastic binary inputs..
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
                Wdec = tf.get_variable('Wdec', shape=(self._hidden_dec.shape[-1], config.dimInput))
                bdec = tf.get_variable('bdec', shape=config.dimInput, initializer=tf.zeros_initializer)
                self._dec = tf.nn.sigmoid(tf.tensordot(self._hidden_dec, Wdec, [[-1], [0]]) + bdec)
                self._loss += BernoulliNLL(self.x, self._dec)
                self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                self._train_step = self._optimizer.minimize(tf.cast(tf.shape(self.x), tf.float32)[-1] * self._loss)
                self._runSession()

"""#########################################################################
Class: _SRNN - .
#########################################################################"""
class gaussSRNN(_SRNN, object):
    pass
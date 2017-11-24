# TODO:
"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the file contains the model description of RNN-RBM.
              ----2017.11.03
#########################################################################"""
from dl4s.TRBM import configRNNRBM
from dl4s.SeqVAE.utility import buildRec
from dl4s.TRBM.RBM import binRBM
import tensorflow as tf

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
            self._train_step = None
            self._ll = None
            self._pll = None
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
    train_function: compute the loss and update the tensor variables.
    input: input - numerical input.
           lrate - <scalar> learning rate.
    output: the loss value.
    #########################################################################"""
    def train_function(self, input, lrate):
        with self._graph.as_default():
            _, loss_value = self._sess.run([self._train_step, self._ll],
                                           feed_dict={self.x: input, self.lr: lrate})
        return loss_value

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
            state = self._rnnCell.zero_state(tf.shape(self.x)[0], dtype=tf.float32)
            d, _ = tf.nn.dynamic_rnn(self._rnnCell, self.x, initial_state=state)
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
            self._pll = self._rbm._pll
            self._logZ = self._rbm.AIS(self._aisRun, self._aisLevel, Batch=tf.shape(self.x)[0],
                                       Seq=tf.shape(self.x)[1])
            self._ll = tf.reduce_mean(-self._rbm.FreeEnergy(self.x) - self._logZ)
            self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self._train_step = self._optimizer.minimize(self._loss)
            self._runSession()


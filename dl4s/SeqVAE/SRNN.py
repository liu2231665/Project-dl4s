# TODO:
"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the file contains the model description of SRNN.
              ----2017.11.15
#########################################################################"""

from tensorflow.python.ops.rnn import _reverse_seq
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

            buildSRNN(self.x, self._graph, config)

"""#########################################################################
Class: _SRNN - .
#########################################################################"""
class binSRNN(_SRNN, object):
    pass

"""#########################################################################
Class: _SRNN - .
#########################################################################"""
class gaussSRNN(_SRNN, object):
    pass
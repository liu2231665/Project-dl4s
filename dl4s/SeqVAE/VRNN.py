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

            buildVRNN(self.x, self._graph, config)

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
            configVRNN
    ):
        super().__init__(configVRNN)

"""#########################################################################
Class: gaussVRNN - the VRNN model for stochastic continuous inputs.
#########################################################################"""
class gaussVRNN(_VRNN, object):
    pass
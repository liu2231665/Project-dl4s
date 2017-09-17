# Autoregressive RNN with logistic outputs.
# Author: Yingru Liu
# Institute: Stony Brook University

import tensorflow as tf
from .utility import inference_net

class binRNN():

    def __init__(
            self,
            unitType,
            dimLayer = []
    ):
        if dimLayer == []:
            raise(ValueError('The structure is empty!'))

        self.x = tf.placeholder(dtype='float32', shape=[None, None, dimLayer[0]])
        self.numLayer = len(dimLayer) - 2
        self.unitType = unitType

        # Build the Inference Network
        hidden = inference_net(unitType, dimLayer, self.x)
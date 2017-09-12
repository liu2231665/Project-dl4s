# Autoregressive RNN with logistic outputs.
# Author: Yingru Liu
# Institute: Stony Brook University

import tensorflow as tf
from collections import OrderedDict

class binRNN():
    def __int__(
            self,
            unitType,
            dimLayer = []
    ):
        if dimLayer == []:
            raise(ValueError('The structure is empty!'))

        self.x = tf.placeholder(dtype='float32', shape=[None, None, dimLayer[0]])
        self.params = OrderedDict()
# TODO:
"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the file contains the model description of VRNN.
              ----2017.11.03
#########################################################################"""

import tensorflow as tf
from .utility import buildSTORN
from . import GaussKL, BernoulliNLL, GaussNLL
from dl4s.tools import get_batches_idx
import numpy as np
import time, os

"""#########################################################################
Class: _VRNN - the hyper abstraction of the VRNN.
#########################################################################"""
class _VRNN(object):
    pass

"""#########################################################################
Class: binSTORN - the VRNN model for stochastic binary inputs.
#########################################################################"""
class binVRNN(_VRNN, object):
    pass

"""#########################################################################
Class: gaussVRNN - the VRNN model for stochastic continuous inputs.
#########################################################################"""
class gaussVRNN(_VRNN, object):
    pass
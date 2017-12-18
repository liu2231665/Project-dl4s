"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: Tools to build an Chain Graph RNN.
              ----2017.12.18
#########################################################################"""
import tensorflow as tf
from dl4s.SeqVAE.utility import buildRec, MLP
from dl4s.TRBM.RBM import bin_ssRBM, mu_ssRBM

"""#########################################################################
Class: CGCell - Basic step of the CGRNN models. 
#########################################################################"""
class CGCell(tf.contrib.rnn.RNNCell):
    """
    __init__: the initialization function.
    input: configSTORN - configuration class in ./utility.
    output: None.
    """
    def __init__(self, config, inputType='binary'):
        self._config = config
        self._dimInput = config.dimInput
        self._inputType = inputType
        self._dimState = config.dimState       # the dimension of stochastic layer
        self._init_scale = config.init_scale   # the initialized scale for the model.
        self.rnnCell = buildRec(self._config.dimRec, self._config.unitType,
                               self._config.init_scale)    # the hidden layer part of the recognition model.

    """
        __call__:
        input: x - the current input with size (batch, frame) where frame = [d_t, a_t]
                   the bounds between them is 0~dimDt, dimDt~end
               state - the previous state of the cells. [bv, bh, gamma, rnnstate]
               scope - indicate the variable scope.
        output: 
        """
    def __call__(self, x, state, scope=None):
        if self._inputType == 'binary':
            bvt, bht = state[0:2]
            self.RBM = bin_ssRBM(dimV=self._config.dimInput, dimH=self._config.dimState,
                                 init_scale=self._config.init_scale,
                                 bh=bht, bv=bvt,
                                 x=x, alphaTrain=self._config.alphaTrain,
                                 muTrain=self._config.muTrain,
                                 k=self._gibbs)
        elif self._inputType == 'continuous':
            bvt, bht, gamma = state[0:3]
            self.RBM = mu_ssRBM(dimV=self._config.dimInput, dimH=self._config.dimState,
                                 init_scale=self._config.init_scale,
                                 bh=bht, bv=bvt, gamma=gamma,
                                 x=x, alphaTrain=self._config.alphaTrain,
                                 muTrain=self._config.muTrain, phiTrain=self._config.phiTrain,
                                 k=self._gibbs)
        else:
            raise ValueError("The input type should be either binary or continuous!!")
        # run the RBM to generate necessary variable.
        rnnstate = state[-1]
        hidden, newState = self.rnnCell(x, rnnstate)
"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: Test code for the binary STORN.
              ----2017.11.08
#########################################################################"""
from dl4s.SeqVAE.STORN import binSTORN
from dl4s.SeqVAE import configSTORN
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X = dict()
    X = np.random.normal(0, 1.0, size=(100, 25, 200))
    Config = configSTORN()
    Config.Opt = 'Momentum'
    Config.dimGen = [200, 500, 300]
    Config.dimReg = [200, 500, 300]
    Config.dimInput = 200
    Config.init_scale = 0.1
    Config.eventPath = './RNN/'
    Config.savePath = './RNN/my-model'

    """
    test training and model operation.
    """
    RNN = binSTORN(Config)
    # test the training function
"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: Test code for the binary VARNN.
              ----2017.11.13
#########################################################################"""
from dl4s.SeqVAE.VRNN import binVRNN
from dl4s.SeqVAE import configVRNN
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X = dict()
    X = np.random.binomial(1, 0.5, size=(100, 250, 200))
    Config = configVRNN()
    Config.Opt = 'SGD'
    Config.dimRec = [200]
    Config.dimForX = [100]
    Config.dimForZ = [100]
    Config.dimInput = 200
    Config.dimState = 200
    Config.init_scale = 0.01
    Config.eventPath = './VRNN/'
    Config.savePath = './VRNN/my-model'

    """
    test training and model operation.
    """
    VRNN = binVRNN(Config)
    pass
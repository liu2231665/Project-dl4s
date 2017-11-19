"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: Test code for the binary SRNN.
              ----2017.11.18
#########################################################################"""
from dl4s.SeqVAE.SRNN import binSRNN
from dl4s.SeqVAE import configSRNN
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X = dict()
    X = np.random.binomial(1, 0.5, size=(100, 40, 200))
    Config = configSRNN()
    Config.Opt = 'SGD'
    mode = 'smooth'
    Config.dimRecD = [400]
    Config.dimRecA = [400]
    Config.dimEnc = [400]
    Config.dimDec = [400]
    Config.dimInput = 200
    Config.dimState = 200
    Config.init_scale = 0.01
    Config.eventPath = './SRNN/'
    Config.savePath = './sRNN/my-model'

    """
    test training and model operation.
    """
    VRNN = binSRNN(Config)
    for i in range(50):
        print("The training ELBO is %f." % VRNN.train_function(input=X, lrate=0.001))
    pass

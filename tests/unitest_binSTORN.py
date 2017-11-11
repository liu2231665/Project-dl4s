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
    X = np.random.binomial(1, 0.5, size=(100, 250, 200))
    Config = configSTORN()
    Config.Opt = 'SGD'
    Config.dimGen = [500]
    Config.dimReg = [500]
    Config.dimInput = 200
    Config.dimState = 500
    Config.init_scale = 0.01
    Config.eventPath = './RNN/'
    Config.savePath = './RNN/my-model'

    """
    test training and model operation.
    """
    RNN = binSTORN(Config)
    # test the training function
    for i in range(100):
        print("The training loss is %f." % RNN.train_function(input=X, lrate=0.1))
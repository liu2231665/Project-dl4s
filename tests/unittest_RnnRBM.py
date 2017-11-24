"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: Test code for the binary RNNRBM.
              ----2017.11.23
#########################################################################"""


from dl4s.TRBM import configRNNRBM
from dl4s.TRBM import binRnnRBM
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X = dict()
    X = np.random.binomial(1, 0.5, size=(100, 250, 200))
    Config = configRNNRBM()
    Config.Opt = 'SGD'
    Config.dimRec = [400]
    Config.dimInput = 200
    Config.dimState = 200
    Config.init_scale = 0.01

    """
    test training and model operation.
    """
    RNNRBM = binRnnRBM(Config)
    for i in range(5):
        print("The training ELBO is %f." % RNNRBM.train_function(input=X, lrate=0.1))

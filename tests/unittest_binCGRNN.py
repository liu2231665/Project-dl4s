"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: Test code for the binary CGRNN.
              ----2017.12.19
#########################################################################"""

from dl4s import configCGRNN as Config
from dl4s import binCGRNN
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    Config.dimInput = 10
    Config.dimState = 25
    Config.dimRec = [50]
    Config.aisLevel = 100
    """
    test training and model operation.
    """
    RNNRBM = binCGRNN(Config)
    X = np.random.binomial(1, 0.5, size=(10, 25, 10))
    for i in range(1):
        print("The training RMSE is %f." % RNNRBM.train_function(input=X, lrate=0.1))
    """
    test the AIS.
    """
    print(RNNRBM.ais_function(input=X))
    pass
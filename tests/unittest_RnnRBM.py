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
    X = np.random.binomial(1, 0.5, size=(10, 250, 200))
    Config = configRNNRBM()
    Config.Opt = 'SGD'
    Config.dimRec = [400]
    Config.dimMlp = [400, 400]
    Config.dimInput = 200
    Config.dimState = 400
    Config.init_scale = 0.01
    Config.Gibbs = 15
    Config.aisRun = 100
    Config.aisLevel = 500
    Config.savePath = './RNNRBM/my-model'

    """
    test training and model operation.
    """
    RNNRBM = binRnnRBM(Config)
    """
    test the full training function.
    """
    X = dict()
    X['train'] = np.random.binomial(1, 0.5, size=(130, 25, 200))
    X['valid'] = np.random.binomial(1, 0.5, size=(130, 25, 200))
    X['test'] = np.random.binomial(1, 0.5, size=(130, 25, 200))
    RNNRBM.full_train(dataset=X, maxEpoch=5, earlyStop=10, batchSize=125, valid_batchSize=25, learning_rate=0.001,
                    saveto='./RNNRBM/results.npz')
    print(RNNRBM.ais_function(input=X['test']))


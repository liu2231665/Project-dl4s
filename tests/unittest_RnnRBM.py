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
    X = np.random.binomial(1, 0.5, size=(10, 250, 200))
    Config = configRNNRBM()
    Config.Opt = 'SGD'
    Config.dimRec = [400]
    Config.dimMlp = [400, 400]
    Config.dimIN = 200
    Config.dimState = 400
    Config.init_scale = 0.001
    Config.Gibbs = 15
    Config.aisRun = 100
    Config.aisLevel = 500
    Config.savePath = None

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
    RNNRBM.full_train(dataset=X, maxEpoch=5, earlyStop=10, batchSize=125, valid_batchSize=25, learning_rate=0.01,
                    saveto=None)
    print(RNNRBM.ais_function(input=X['test']))

    X = np.random.binomial(1, 0.5, size=(100, 40, 200))
    """
    test the generating sample.
    """
    samples = RNNRBM.generate(numSteps=40)
    plt.figure(1)
    plt.imshow(samples, cmap='binary')

    """
    test the feature embedding.
    """
    feature = RNNRBM.embed(X[0])
    plt.figure(2)
    plt.imshow(feature[0], cmap='jet')

    """
        test the reconstruction.
        """
    re = RNNRBM.reconstruct(X[0:2])
    plt.figure(3)
    plt.imshow(re[0], cmap='jet')
    plt.show()


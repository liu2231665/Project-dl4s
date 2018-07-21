"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: Test code for the Gaussian RNNRBM.
              ----2017.11.29
#########################################################################"""


from dl4s.TRBM import configRNNRBM
from dl4s.TRBM import gaussRnnRBM
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X = dict()
    X = np.random.normal(0, 1.0, size=(100, 25, 200))
    Config = configRNNRBM()
    Config.Opt = 'SGD'
    Config.dimRec = [400]
    Config.dimMlp = [400, 400]
    Config.dimIN = 200
    Config.dimState = 400
    Config.init_scale = 0.01
    Config.Gibbs = 15
    Config.aisLevel = 20
    Config.aisRun = 20
    Config.savePath = None

    """
    test training and model operation.
    """
    RNNRBM = gaussRnnRBM(Config)
    for i in range(50):
        print("The training ELBO is %f." % RNNRBM.train_function(input=X, lrate=0.001))

    """
    test the generating sample.
    """
    samples = RNNRBM.generate(numSteps=40)
    plt.figure(1)
    plt.imshow(samples, cmap='jet')
    """
    test the encoder model.
    """
    plt.figure(2)
    mu= RNNRBM.reconstruct(X)
    plt.imshow(mu[0, :, :], cmap='jet')
    plt.show()
    """
    test saving and restoring model.
    """
    RNNRBM.saveModel()
    loadPath = None
    RNNRBM.loadModel(loadPath)
    for i in range(10):
        print(RNNRBM.train_function(input=X, lrate=0.01))

    """
    test multi-graph (One RNN instant = one graph).
    """
    RNNRBM2 = gaussRnnRBM(Config)

    """
    test the full training function.
    """
    X = dict()
    X['train'] = np.random.binomial(1, 0.5, size=(130, 25, 200))
    X['valid'] = np.random.binomial(1, 0.5, size=(130, 25, 200))
    X['test'] = np.random.binomial(1, 0.5, size=(130, 25, 200))
    RNNRBM.full_train(dataset=X, maxEpoch=5, earlyStop=10, batchSize=125, valid_batchSize=25, learning_rate=0.001,
                    saveto=None)


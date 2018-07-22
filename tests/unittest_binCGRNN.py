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
    Config.dimIN = 50
    Config.Opt = 'Adam'
    Config.dimRec = [100]
    Config.dimMlp = [100, 100]
    Config.dimState = 100
    Config.init_scale = 0.01
    Config.Gibbs = 1
    Config.aisLevel = 20
    Config.aisRun = 20
    Config.W_Norm = True
    Config.savePath = None
    Config.mode = 'full'
    """
    test training and model operation.
    """
    CGRNN = binCGRNN(Config)
    X = np.random.binomial(1, 0.5, size=(100, 25, 50))
    for i in range(100):
        print("The training error is %f." % CGRNN.train_function(input=X, lrate=0.1))
    print(CGRNN.val_function(X))
    """
    test the AIS.
    """
    print(CGRNN.ais_function(input=X))

    """
        test the generating sample.
        """
    print("The valid RMSE is %f." % CGRNN.val_function(input=X))
    samples = CGRNN.generate(numSteps=40)
    plt.figure(1)
    plt.imshow(samples, cmap='jet')
    """
    test the encoder model.
    """
    plt.figure(2)
    mu = CGRNN.embed(X, 'sparse')
    plt.imshow(mu[0, :, :], cmap='jet')
    plt.show()

    """
    test the full training function.
    """
    X = dict()
    X['train'] = np.random.binomial(1, 0.5, size=(130, 25, 50))
    X['valid'] = np.random.binomial(1, 0.5, size=(130, 25, 50))
    X['test'] = np.random.binomial(1, 0.5, size=(130, 25, 50))
    CGRNN.full_train(dataset=X, maxEpoch=5, earlyStop=10, batchSize=125, valid_batchSize=25, learning_rate=0.01,
                      saveto=None)
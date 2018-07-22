"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: Test code for the binary ssRNNRBM.
              ----2017.12.16
#########################################################################"""


from dl4s.TRBM import configssRNNRBM
from dl4s.TRBM import binssRNNRBM
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X = dict()
    X = np.random.binomial(1, 0.5, size=(10, 250, 50))
    Config = configssRNNRBM()
    Config.Opt = 'Adam'
    Config.dimRec = [100]
    Config.dimMlp = [100, 100]
    Config.dimIN = 50
    Config.dimState = 100
    Config.init_scale = 0.01
    Config.Gibbs = 1
    Config.aisLevel = 20
    Config.aisRun = 20
    Config.W_Norm = True
    Config.savePath = None

    """
    test training and model operation.
    """
    RNNRBM = binssRNNRBM(Config)
    for i in range(10):
        print("The training RMSE is %f." % RNNRBM.train_function(input=X, lrate=0.1))
    print(RNNRBM.val_function(X))

    """
    test the generating sample.
    """
    print("The valid RMSE is %f." % RNNRBM.val_function(input=X))
    samples = RNNRBM.generate(numSteps=40)
    plt.figure(1)
    plt.imshow(samples, cmap='jet')
    """
    test the encoder model.
    """
    plt.figure(2)
    mu= RNNRBM.embed(X)
    plt.imshow(mu[0, :, :], cmap='jet')
    plt.show()

    """
    test the full training function.
    """
    X = dict()
    X['train'] = np.random.binomial(1, 0.5, size=(130, 25, 50))
    X['valid'] = np.random.binomial(1, 0.5, size=(130, 25, 50))
    X['test'] = np.random.binomial(1, 0.5, size=(130, 25, 50))
    RNNRBM.full_train(dataset=X, maxEpoch=5, earlyStop=10, batchSize=125, valid_batchSize=25, learning_rate=0.01,
                    saveto=None)


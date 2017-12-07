"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: Test code for the ssRNNRBM.
              ----2017.12.05
#########################################################################"""


from dl4s.TRBM import configssRNNRBM
from dl4s.TRBM import ssRNNRBM
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X = dict()
    X = np.random.normal(0, 1.0, size=(100, 25, 50))
    Config = configssRNNRBM()
    Config.Opt = 'Adam'
    Config.dimRec = [100]
    Config.dimMlp = [100, 100]
    Config.dimInput = 50
    Config.dimState = 100
    Config.init_scale = 0.01
    Config.Gibbs = 1
    Config.aisLevel = 20
    Config.aisRun = 20
    Config.W_Norm = True
    Config.savePath = './RNNRBM/my-model'

    """
    test training and model operation.
    """
    RNNRBM = ssRNNRBM(Config)
    for i in range(10):
        print("The training RMSE is %f." % RNNRBM.train_function(input=X, lrate=0.005))
    """
    test precision and covariance operation.
    """
    pre = RNNRBM.cov_function(X)
    cov = RNNRBM.pre_function(X)
    print(RNNRBM.ais_function(X))

    """
    test the generating sample.
    """
    print("The valid RMSE is %f." % RNNRBM.val_function(input=X))
    samples = RNNRBM.gen_function(numSteps=40)
    plt.figure(1)
    plt.imshow(samples, cmap='jet')
    """
    test the encoder model.
    """
    plt.figure(2)
    mu= RNNRBM.hidden_function(X)
    plt.imshow(mu[0, :, :], cmap='jet')
    plt.figure(3)
    plt.imshow(pre[0, 5, :, :], cmap='jet')
    plt.figure(4)
    plt.imshow(cov[0, 5, :, :], cmap='jet')
    plt.show()

    """
    test the full training function.
    """
    X = dict()
    X['train'] = np.random.binomial(1, 0.5, size=(130, 25, 50))
    X['valid'] = np.random.binomial(1, 0.5, size=(130, 25, 50))
    X['test'] = np.random.binomial(1, 0.5, size=(130, 25, 50))
    RNNRBM.full_train(dataset=X, maxEpoch=5, earlyStop=10, batchSize=125, valid_batchSize=25, learning_rate=0.001,
                    saveto='./RNNRBM/results.npz')


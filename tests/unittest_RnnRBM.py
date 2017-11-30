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
    Config.savePath = './RNNRBM/my-model'

    """
    test training and model operation.
    """
    RNNRBM = binRnnRBM(Config)
    for i in range(50):
        print("The training ELBO is %f." % RNNRBM.train_function(input=X, lrate=0.01))

    """
    test the generating sample.
    """
    print("The valid ELBO is %f." % RNNRBM.val_function(input=X))
    samples = RNNRBM.gen_function(numSteps=40)
    plt.figure(1)
    plt.imshow(samples, cmap='binary')
    """
    test the encoder model.
    """
    plt.figure(2)
    mu= RNNRBM.hidden_function(X)
    plt.imshow(mu[0, :, :], cmap='jet')
    plt.show()
    """
    test saving and restoring model.
    """
    RNNRBM.saveModel()
    loadPath = './RNNRBM/my-model'
    RNNRBM.loadModel(loadPath)
    for i in range(10):
        print(RNNRBM.train_function(input=X, lrate=0.01))

    """
    test multi-graph (One RNN instant = one graph).
    """
    RNNRBM2 = binRnnRBM(Config)

    """
    test the full training function.
    """
    X = dict()
    X['train'] = np.random.binomial(1, 0.5, size=(130, 25, 200))
    X['valid'] = np.random.binomial(1, 0.5, size=(130, 25, 200))
    X['test'] = np.random.binomial(1, 0.5, size=(130, 25, 200))
    RNNRBM.full_train(dataset=X, maxEpoch=5, earlyStop=10, batchSize=125, valid_batchSize=25, learning_rate=0.001,
                    saveto='./RNNRBM/results.npz')


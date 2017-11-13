"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: Test code for the Gaussian STORN.
              ----2017.11.13
#########################################################################"""

from dl4s import gaussSTORN
from dl4s import configSTORN
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X = dict()
    X = np.random.normal(0, 1.0, size=(100, 25, 200))
    Config = configSTORN()
    Config.Opt = 'Adam'
    Config.dimGen = [500]
    Config.dimReg = [500]
    Config.dimInput = 200
    Config.dimState = 500
    Config.init_scale = 0.01
    Config.eventPath = './STORN/'
    Config.savePath = './STORN/my-model'

    """
    test training and model operation.
    """
    STORN = gaussSTORN(Config)
    for i in range(5):
        print("The training ELBO is %f." % STORN.train_function(input=X, lrate=0.01))

    """
    test the generating sample.
    """
    print("The valid ELBO is %f." % STORN.val_function(input=X))
    samples = STORN.gen_function(numSteps=40)
    plt.figure(1)
    plt.imshow(samples, cmap='jet')

    """
    test the recognition model.
    """
    plt.figure(2)
    mu, sigma2 = STORN.recognitionOutput(X)
    plt.subplot(211)
    plt.imshow(mu[0, :, :], cmap='jet')
    plt.subplot(212)
    plt.imshow(sigma2[0, :, :], cmap='jet')
    plt.show()

    """
    test saving and restoring model.
    """
    STORN.saveModel()
    loadPath = './STORN/my-model'
    STORN.loadModel(loadPath)
    for i in range(10):
        print(STORN.train_function(input=X, lrate=0.1))

    """
    test multi-graph (One RNN instant = one graph).
    """
    RNN2 = gaussSTORN(Config)

    """
    test saving events.
    """
    STORN.saveEvent()

    """
    test the full training function.
    """
    X = dict()
    X['train'] = np.random.normal(1, 0.5, size=(130, 25, 200))
    X['valid'] = np.random.normal(1, 0.5, size=(130, 25, 200))
    X['test'] = np.random.normal(1, 0.5, size=(130, 25, 200))
    STORN.full_train(dataset=X, maxEpoch=5, earlyStop=10, batchSize=20, learning_rate=0.1, saveto='./STORN/results.npz')
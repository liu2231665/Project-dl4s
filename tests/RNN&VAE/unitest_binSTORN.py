"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: Test code for the binary STORN.
              ----2017.11.08
#########################################################################"""
from dl4s.SeqVAE.STORN import binSTORN
from dl4s.SeqVAE import configSTORN
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X = dict()
    X = np.random.binomial(1, 0.5, size=(100, 250, 200))
    Config = configSTORN()
    Config.Opt = 'SGD'
    Config.dimGen = [200]
    Config.dimReg = [200]
    Config.dimInput = 200
    Config.dimState = 200
    Config.init_scale = 0.01
    Config.eventPath = './STORN/'
    Config.savePath = './STORN/my-model'

    """
    test training and model operation.
    """
    STORN = binSTORN(Config)
    for i in range(5):
        print("The training ELBO is %f." % STORN.train_function(input=X, lrate=0.1))

    """
    test the generating sample.
    """
    print("The valid ELBO is %f." % STORN.val_function(input=X))
    samples = STORN.gen_function(numSteps=40)
    plt.figure(1)
    plt.imshow(samples, cmap='binary')

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
    RNN2 = binSTORN(Config)

    """
    test saving events.
    """
    STORN.saveEvent()

    """
    test the full training function.
    """
    X = dict()
    X['train'] = np.random.binomial(1, 0.5, size=(130, 25, 200))
    X['valid'] = np.random.binomial(1, 0.5, size=(130, 25, 200))
    X['test'] = np.random.binomial(1, 0.5, size=(130, 25, 200))
    STORN.full_train(dataset=X, maxEpoch=5, earlyStop=10,  batchSize=125, learning_rate=0.1, saveto='./STORN/results.npz')

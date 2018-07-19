"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: Test code for the binary SRNN.
              ----2017.11.18
#########################################################################"""
from dl4s.SeqVAE.SRNN import binSRNN
from dl4s.SeqVAE import configSRNN
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X = np.random.binomial(1, 0.5, size=(100, 250, 200))
    Config = configSRNN()
    Config.Opt = 'SGD'
    mode = 'smooth'
    Config.dimRecD = [400]
    Config.dimRecA = [400]
    Config.dimEnc = [400]
    Config.dimDec = [400]
    Config.dimIN = 200
    Config.dimState = 200
    Config.init_scale = 0.01
    Config.eventPath = None
    Config.savePath = None

    """
    test training and model operation.
    """
    SRNN = binSRNN(Config)
    for i in range(50):
        print("The training ELBO is %f." % SRNN.train_function(input=X, lrate=0.01))
    """
    test the generating sample.
    """
    print("The valid ELBO is %f." % SRNN.val_function(input=X))
    samples = SRNN.generate(numSteps=40)
    recon = SRNN.reconstruct(samples.reshape(1, 40, 200))[0]
    plt.figure(1)
    plt.imshow(samples, cmap='binary')
    plt.figure(2)
    plt.imshow(recon, cmap='binary')

    """
    test the encoder model.
    """
    plt.figure(3)
    mu, std = SRNN.encoder(X)
    plt.subplot(211)
    plt.imshow(mu[0, :, :], cmap='jet')
    plt.subplot(212)
    plt.imshow(std[0, :, :], cmap='jet')
    plt.show()


    """
    test saving and restoring model.
    """
    SRNN.saveModel()
    loadPath = None
    SRNN.loadModel(loadPath)
    for i in range(10):
        print(SRNN.train_function(input=X, lrate=0.001))

    """
    test multi-graph (One RNN instant = one graph).
    """
    RNN2 = binSRNN(Config)

    """
    test saving events.
    """
    SRNN.saveEvent()

    """
    test the full training function.
    """
    X = dict()
    X['train'] = np.random.binomial(1, 0.5, size=(130, 25, 200))
    X['valid'] = np.random.binomial(1, 0.5, size=(130, 25, 200))
    X['test'] = np.random.binomial(1, 0.5, size=(130, 25, 200))
    SRNN.full_train(dataset=X, maxEpoch=5, earlyStop=10, batchSize=125, learning_rate=0.001,
                     saveto=None)



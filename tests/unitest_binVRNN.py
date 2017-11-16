"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: Test code for the binary VARNN.
              ----2017.11.13
#########################################################################"""
from dl4s.SeqVAE.VRNN import binVRNN
from dl4s.SeqVAE import configVRNN
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X = dict()
    X = np.random.binomial(1, 0.5, size=(100, 40, 200))
    Config = configVRNN()
    Config.Opt = 'SGD'
    Config.dimRec = [200]
    Config.dimForX = [100]
    Config.dimForZ = [100]
    Config.dimInput = 200
    Config.dimState = 200
    Config.init_scale = 0.01
    Config.eventPath = './VRNN/'
    Config.savePath = './VRNN/my-model'

    """
    test training and model operation.
    """
    VRNN = binVRNN(Config)
    for i in range(50):
        print("The training ELBO is %f." % VRNN.train_function(input=X, lrate=0.01))

    """
    test the generating sample.
    """
    print("The valid ELBO is %f." % VRNN.val_function(input=X))
    samples = VRNN.gen_function(numSteps=40)
    plt.figure(1)
    plt.imshow(samples, cmap='binary')

    """
    test the encoder model.
    """
    plt.figure(2)
    mu, std = VRNN.encoder(X)
    plt.subplot(211)
    plt.imshow(mu[0, :, :], cmap='jet')
    plt.subplot(212)
    plt.imshow(std[0, :, :], cmap='jet')
    plt.show()

    """
    test saving and restoring model.
    """
    VRNN.saveModel()
    loadPath = './VRNN/my-model'
    VRNN.loadModel(loadPath)
    for i in range(10):
        print(VRNN.train_function(input=X, lrate=0.01))

    """
    test multi-graph (One RNN instant = one graph).
    """
    RNN2 = binVRNN(Config)

    """
    test saving events.
    """
    VRNN.saveEvent()

    """
    test the full training function.
    """
    X = dict()
    X['train'] = np.random.binomial(1, 0.5, size=(130, 25, 200))
    X['valid'] = np.random.binomial(1, 0.5, size=(130, 25, 200))
    X['test'] = np.random.binomial(1, 0.5, size=(130, 25, 200))
    VRNN.full_train(dataset=X, maxEpoch=5, earlyStop=10, batchSize=125, learning_rate=0.01,
                     saveto='./VRNN/results.npz')


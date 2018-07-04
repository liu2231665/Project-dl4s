"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: Test code for the binary RNN.
              ----2017.11.02
#########################################################################"""


from dl4s.autoregRnn import binRNN
from dl4s.autoregRnn import config
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X = dict()
    X = np.random.binomial(1, 0.5, size=(40, 25, 100))
    Config = config()
    Config.dimLayer = [100, 500, 100]
    Config.eventPath = None
    Config.savePath = None

    """
    test training and model operation.
    """
    RNN = binRNN(Config)
    # test the training function
    for i in range(10):
        print("The training loss is %f." % RNN.train_function(input=X, lrate=0.1))
    # test the generating sample.
    print("The valid loss is %f." % RNN.val_function(input=X))
    samples = RNN.generate(numSteps=40)
    imgplot = plt.imshow(samples, cmap='binary')
    plt.show()

    """
    test saving and restoring model.
    """
    RNN.saveModel()
    loadPath = None
    RNN.loadModel(loadPath)
    for i in range(10):
        print(RNN.train_function(input=X, lrate=0.1))

    """
    test multi-graph (One RNN instant = one graph).
    """
    RNN2 = binRNN(Config)

    """
    test saving events.
    """
    RNN.saveEvent()

    """
    test the full training function.
    """
    X = dict()
    X['train'] = np.random.binomial(1, 0.5, size=(130, 25, 100))
    X['valid'] = np.random.binomial(1, 0.5, size=(130, 25, 100))
    X['test'] = np.random.binomial(1, 0.5, size=(130, 25, 100))
    RNN.full_train(dataset=X, maxEpoch=5, earlyStop=10,  batchSize=125, learning_rate=0.1, saveto=None)
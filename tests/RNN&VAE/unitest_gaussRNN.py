"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: Test code for the Gaussian RNN.
              ----2017.11.02
#########################################################################"""

from dl4s.autoregRnn import gaussRNN
from dl4s.autoregRnn import config
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X = dict()
    X = np.random.normal(0, 1.0, size=(100, 25, 200))
    Config = config()
    Config.Opt = 'Momentum'
    Config.dimLayer = [500]
    Config.dimIN = 200
    Config.init_scale = 0.1
    Config.eventPath = './RNN/'
    Config.savePath = './RNN/my-model'

    """
    test training and model operation.
    """
    RNN = gaussRNN(Config)
    # test the training function
    for i in range(100):
        print("The training loss is %f." % RNN.train_function(input=X, lrate=0.1))
    # test the saving and restoring
    print("The valid loss is %f." % RNN.val_function(input=X))
    samples = RNN.generate(numSteps=100)
    feature = RNN.embed(samples)
    imgplot = plt.imshow(samples, cmap='jet')
    plt.show()

    """
    test saving and restoring model.
    """
    RNN.saveModel()
    loadPath = './RNN/my-model'
    RNN.loadModel(loadPath)
    for i in range(10):
        print(RNN.train_function(input=X, lrate=0.1))

    """
    test multi-graph (One RNN instant = one graph).
    """
    RNN2 = gaussRNN(Config)

    """
    test saving events.
    """
    RNN.saveEvent()

    """
    test the full training function.
    """
    X = dict()
    X['train'] = np.random.normal(1, 0.5, size=(130, 25, 200))
    X['valid'] = np.random.normal(1, 0.5, size=(130, 25, 200))
    X['test'] = np.random.normal(1, 0.5, size=(130, 25, 200))
    RNN.full_train(dataset=X, maxEpoch=5, earlyStop=10, batchSize=20, learning_rate=0.001, saveto='./RNN/results.npz')
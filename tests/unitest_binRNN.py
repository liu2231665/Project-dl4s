from dl4stochastic.autoregRnn import binRNN
from dl4stochastic.autoregRnn import config
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X = dict()
    X = np.random.binomial(1, 0.5, size=(40, 25, 100))
    Config = config()
    Config.max_steps = 26
    Config.batch_size = 40
    Config.dimLayer = [100, 500, 100]
    Config.eventPath = './'
    Config.savePath = './my-model'

    """
    test training and model operation.
    """
    RNN = binRNN(Config)
    # # test the training function
    # for i in range(10):
    #     print("The training loss is %f." % RNN.train_function(input=X, lrate=1.0))
    # # test the saving and restoring
    # print("The valid loss is %f." % RNN.val_function(input=X))
    # samples = RNN.gen_function(numSteps=40)
    # imgplot = plt.imshow(samples)
    # plt.show()
    #
    # """
    # test saving and restoring model.
    # """
    # RNN.saveModel()
    # loadPath = './my-model'
    # RNN.loadModel(loadPath)
    # for i in range(10):
    #     print(RNN.train_function(input=X, lrate=1.0))
    #
    # """
    # test multi-graph (One RNN instant = one graph).
    # """
    # RNN2 = binRNN(Config)
    #
    # """
    # test saving events.
    # """
    # RNN.saveEvent()

    """
    test the full training function.
    """
    X = dict()
    X['train'] = np.random.binomial(1, 0.5, size=(130, 25, 100))
    X['valid'] = np.random.binomial(1, 0.5, size=(130, 25, 100))
    X['test'] = np.random.binomial(1, 0.5, size=(130, 25, 100))
    RNN.full_train(dataset=X, max_epoch=100, earlyStop=10, learning_rate=0.1)
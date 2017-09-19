from dl4stochastic.autoregRnn import binRNN
from dl4stochastic.autoregRnn import config
import numpy as np

if __name__ == '__main__':
    X = np.random.binomial(1, 0.5, size=(40, 25, 100))
    Config = config()
    Config.max_steps = 26
    Config.batch_size = 40
    Config.dimLayer = [100, 500, 100]
    Config.eventPath = './'
    RNN = binRNN(Config)

    for i in range(500):
        print(RNN.train_function(input=X, lrate=1.0))
    samples = RNN.gen_function(numSteps=4)
    pass
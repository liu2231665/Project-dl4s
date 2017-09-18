from dl4stochastic.autoregRnn import binRNN
from dl4stochastic.autoregRnn import config
import numpy as np

if __name__ == '__main__':
    X = np.random.rand(40, 25, 200)
    Config = config()
    Config.max_steps = 10
    Config.dimLayer = [100, 200, 100]
    RNN = binRNN(Config)
    pass
from dl4stochastic.autoregRnn import binRNN
import numpy as np

if __name__ == '__main__':
    X = np.random.rand(40, 25, 200)
    RNN = binRNN(unitType='LSTM', dimLayer=[200, 100, 200])
    pass
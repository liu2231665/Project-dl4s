"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: Test code for the continuous CGRNN.
              ----2017.12.24
#########################################################################"""

from dl4s import configCGRNN as Config
from dl4s import gaussCGRNN
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    Config.dimIN = 10
    Config.dimState = 25
    Config.dimRec = [50]
    """
    test training and model operation.
    """
    RNNRBM = gaussCGRNN(Config)
    X = np.random.normal(0, 1.0, size=(10, 25, 10))
    for i in range(50):
        print("The training RMSE is %f." % RNNRBM.train_function(input=X, lrate=0.1))
    """
    test the generating sample.
    """
    print("The valid RMSE is %f." % RNNRBM.val_function(input=X))
    samples = RNNRBM.generate(numSteps=40)
    plt.figure(1)
    plt.imshow(samples, cmap='jet')
    """
    test the encoder model.
    """
    plt.figure(2)
    mu = RNNRBM.embed(X, 'sparse')
    plt.imshow(mu[0, :, :], cmap='jet')
    plt.show()

"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: Test code for the non-temporal binary RBM.
              ----2017.11.20
#########################################################################"""

import numpy as np
import matplotlib.pyplot as plt
from dl4s import binRBM

if __name__ == '__main__':
    X = dict()
    X = np.random.binomial(1, 0.5, size=(400, 100))
    dimV = 100
    dimH = 200
    init_scale = 0.01

    """
    test training and model operation.
    """
    RBM = binRBM(dimV, dimH, init_scale)
    pass
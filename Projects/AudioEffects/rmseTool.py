"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the file contain the tools to compute the reconstructed error
              of the models given by RMSE.
              ----2017.11.07
#########################################################################"""

import numpy as np
from dl4s.tools import get_batches_idx

"""#########################################################################
Function: rmseRNN - compute average RMSE of reconstructed samples for RNN
input: RNN - the well-trained RNN model.
       testSet - the test set.
       batches_idx - the batches.
output: RMSE - the average RMSE per frame over the test set.
#########################################################################"""
def rmseRNN(RNN, testSet, batchSize):
    RMSE = []
    batches = get_batches_idx(len(testSet), batchSize)
    for Idx in batches:
        x = testSet[Idx.tolist()]           # get the batch of input sequence.
        ave, _ = RNN.output_function(x)       # compute the probability of x. [batch, length, frame]
        rmse = (x - ave)**2
        rmse = rmse.sum(-1)
        RMSE.append(rmse.mean()*x.shape[0])
    #
    RMSE = np.asarray(RMSE).sum() / len(testSet)
    return np.sqrt(RMSE)
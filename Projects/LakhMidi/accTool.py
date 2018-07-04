"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the file contain the tools to compute the accuracy of the
              piano-rolls prediction, which is given by
                ACC = TP / (TP + FP + FN + 1e-8),    --- 1e-8 to avoid nan
              where
                TP - the number of predicted '1' & ground-true '1';
                FP - the number of predicted '1' & ground-true '0';
                FN - the difference between the counts of predicted '0'
                     and the counts of ground-true '0'.
              For further details, please refer to the paper [*].
              ----2017.11.03
              -> [*] A discriminative model for polyphonic piano
                     transcription.
                     https://link.springer.com/article/10.1155/2007/48317
#########################################################################"""

import numpy as np
from dl4s.cores.tools import get_batches_idx

"""#########################################################################
Function: accRNN - compute the average accuracy of piano-rolls for RNN
input: RNN - the well-trained RNN model.
       testSet - the test set.
       batches_idx - the batches.
       Sample - the number of samples.
output: ACC - the average accuracy per frame over the test set.
#########################################################################"""
def accRNN(RNN, testSet, batchSize, Sample=50):
    ACC = []
    batches = get_batches_idx(len(testSet), batchSize)
    for Idx in batches:
        x = testSet[Idx.tolist()]           # get the batch of input sequence.
        prob = RNN.output_function(x)       # compute the probability of x. [batch, length, frame]
        acc = []
        for i in range(Sample):
            sample = np.random.binomial(1, prob)
            TP = np.asarray((x == 1) & (sample == 1), 'float32').sum(axis=-1)
            FP = np.asarray((x == 0) & (sample == 1), 'float32').sum(axis=-1)
            FN = np.asarray((x == 1) & (sample == 0), 'float32').sum(axis=-1)
            temp = TP / (FN + FP + TP + 1e-8)
            acc.append(temp.mean())
        acc = np.asarray(acc)
        ACC.append(acc.mean()*x.shape[0])
    #
    ACC = np.asarray(ACC).sum() / len(testSet)
    return ACC

"""#########################################################################
Function: accRBM - compute the average accuracy and log-likelihood of 
                   piano-rolls for RNNRBM
input: RBM - the well-trained RNN-RBM model.
       testSet - the test set.
       batches_idx - the batches.
       Sample - the number of samples.
output: ACC - the average accuracy per frame over the test set.
#########################################################################"""
def accRBM(RBM, testSet, batchSize, Sample=25):
    ACC = []
    NLL = []
    batches = get_batches_idx(len(testSet), batchSize)
    for i in range(len(batches)):
        print("[%d/%d]" % (i, len(batches)))
        Idx = batches[i]
        x = testSet[Idx.tolist()]           # get the batch of input sequence.
        acc = []
        for i in range(Sample):
            sample = RBM.gen_function(x=x, gibbs=RBM._gibbs)
            TP = np.asarray((x == 1) & (sample == 1), 'float32').sum(axis=-1)
            FP = np.asarray((x == 0) & (sample == 1), 'float32').sum(axis=-1)
            FN = np.asarray((x == 1) & (sample == 0), 'float32').sum(axis=-1)
            temp = TP / (FN + FP + TP + 1e-8)
            acc.append(temp.mean())
        acc = np.asarray(acc)
        ACC.append(acc.mean()*x.shape[0])
        NLL.append(RBM.ais_function(x)*x.shape[0])
    #
    ACC = np.asarray(ACC).sum() / len(testSet)
    NLL = np.asarray(NLL).sum() / len(testSet)
    return ACC, NLL
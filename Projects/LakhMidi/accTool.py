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

"""#########################################################################
Function: accRNN - compute the average accuracy of piano-rolls for RNN
input: RNN - the well-trained RNN model.
       testSet - the test set.
       batches_idx - the batches.
output: ACC - the average accuracy per frame over the test set.
#########################################################################"""
def accRNN(RNN, testSet, batches):
    ACC = []
    for Idx in batches:
        x = testSet[Idx.tolist()]           # get the batch of input sequence.
        prob = RNN.output_function(x)       # compute the probability of x. [batch, length, frame]
        TP = np.asarray((x == 1) & (prob >= 0.5), 'float32').sum(axis=-1)
        FP = np.asarray((x == 0) & (prob >= 0.5), 'float32').sum(axis=-1)
        FN = np.asarray((x == 1) & (prob < 0.5), 'float32').sum(axis=-1)
        acc = TP / (FN + FP + TP)
        ACC.append(acc.mean()*x.shape[0])
    ACC = np.asarray(ACC).sum() / len(testSet)
    return ACC
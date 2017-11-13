"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: Common Tools that will be used by all the models.
              ----2017.11.01
#########################################################################"""

import numpy as np
import tensorflow as tf


"""#########################################################################
get_minibatches_idx: Used to shuffle the dataset at each iteration.
input: len - the length the dataset section.
       batch_size - the batch size.
       shuffle - bool indicating whether shuffle the idx.
#########################################################################"""
def get_batches_idx(len, batch_size, shuffle=True):

    idx_list = np.arange(len, dtype="int32")
    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(len // batch_size):
        batch = idx_list[minibatch_start:minibatch_start + batch_size]
        batch = np.sort(batch)
        minibatches.append(batch)
        minibatch_start += batch_size

    if (minibatch_start != len):
        # Make a minibatch out of what is left
        batch = idx_list[minibatch_start:]
        batch = np.sort(batch)
        minibatches.append(batch)

    return minibatches

"""#########################################################################
BernoulliNLL: function to compute the negative log-likelihood of Bernoulli
              distribution.
input: x - network input indicated by <tensor placeholder>. 
       P - the probability of 1.
output: nll - a tensor representing the NLL per bit.
#########################################################################"""
def BernoulliNLL(x, P):
    nll = x * tf.log(P+1e-8) + (1 - x) * tf.log(1-P+1e-8)
    return -tf.reduce_mean(nll)

"""#########################################################################
GaussNLL: function to compute the negative log-likelihood of Gaussian 
          distribution with a diagonal covariance matrix.
input: x - network input indicated by <tensor placeholder>. 
       mean - mean of the Gaussian distribution computed by the graph.
       sigma - variance of the Gaussian distribution computed by the graph.
output: nll - a tensor representing the NLL per bit.
#########################################################################"""
def GaussNLL(x, mean, sigma):
    nll = 0.5*tf.reduce_mean(tf.div(tf.square(x-mean), sigma) + tf.log(sigma)) + 0.5*tf.log(2*np.pi)
    return nll

"""#########################################################################
GaussKL: function to compute KL divergence of  two Gaussian distributions
         with a diagonal covariance matrices.
input: meanP - mean of the Gaussian distribution "P"
       sigmaP - variance of the Gaussian distribution "P".
       meanQ - mean of the Gaussian distribution "P"
       sigmaQ - variance of the Gaussian distribution "P".
output: kl - a tensor representing the KL divergence per bit.
#########################################################################"""
def GaussKL(meanP, sigmaP, meanQ, sigmaQ):
    term1 = tf.log(sigmaQ + 1e-8) - tf.log(sigmaP + 1e-8)
    term2 = tf.div(sigmaP + (meanP - meanQ)**2, sigmaQ + 1e-8)
    return 0.5 * tf.reduce_mean(term1 + term2) - 0.5
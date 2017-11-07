"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: Tools to build an RNN.
              ----2017.11.01
#########################################################################"""

import tensorflow as tf
import numpy as np

"""#########################################################################
Class: config - Basic configuration of the auto-regressive RNN.
#########################################################################"""
class config(object):
    """
    Elements outside the __init__ method are static elements.
    Elements inside the __init__ method are elements of the object.
    ----from Stackoverflow(https://stackoverflow.com/questions/9056957/correct-way-to-define-class-variables-in-python).
    """
    unitType = 'LSTM'           # <string> the type of hidden units(LSTM/GRU/Tanh).
    dimLayer = []               # <scalar list> the size of each layers [input, hiddens, output].
    init_scale = 0.1            # <scalar> the initialized scales of the weight.
    float = 'float32'           # <string> the type of float.
    Opt = 'SGD'                 # <string> the optimization method.
    savePath = None             # <string/None> the path to save the model.
    eventPath = None            # <string/None> the path to save the events for visualization.
    loadPath = None             # <string/None> the path to load the model.

#
"""#########################################################################
hidden_net: function to build the hidden layers of the RNN
input: x - network input indicated by <tensor placeholder>. 
       Config - configuration class.
output: cells - tensorflow symbol for the hidden layers of the multi-layer RNN.
        outputs.reshape - the output of last hidden layer.
        initializer - the initializer that may be used later.
#########################################################################"""
def hidden_net(
        x,
        graph,
        Config=config(),
):
    # get the number of hidden layers.
    numLayer = len(Config.dimLayer) - 2
    # define the initializer.
    with graph.as_default():
        initializer = tf.random_uniform_initializer(-Config.init_scale, Config.init_scale)
        # <list> stacks of the hidden layers.
        layers = []
        for i in range(numLayer):
            tf.variable_scope('hidden_' + str(i + 1), initializer=initializer)
            if Config.unitType == 'LSTM':
                layers.append(tf.nn.rnn_cell.LSTMCell(num_units=Config.dimLayer[i + 1]))
            elif Config.unitType == 'GRU':
                layers.append(tf.nn.rnn_cell.GRUCell(num_units=Config.dimLayer[i + 1]))
            else:
                layers.append(tf.nn.rnn_cell.BasicRNNCell(num_units=Config.dimLayer[i + 1]))
        cells = tf.contrib.rnn.MultiRNNCell(layers, state_is_tuple=True)
        state = cells.zero_state(tf.shape(x)[0], Config.float)

        #output: [batch_size, max_time, cell.output_size]
        outputs, _ = tf.nn.dynamic_rnn(cells, x, initial_state=state)
    return cells, outputs, initializer

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
CovGaussNLL: function to compute the negative log-likelihood of Gaussian 
          distribution with a non-diagonal covariance matrix.
input: x - network input indicated by <tensor placeholder>. 
       mean - mean of the Gaussian distribution computed by the graph.
              [batchSize, Time, frameSize]
       sqrtSig - square root of covariance matrix of the Gaussian distribution 
                 computed by the graph. [batchSize, Time, frameSize, frameSize]
output: nll - a tensor representing the NLL per bit.
#########################################################################"""
def CovGaussNLL(x, mean, sqrtSig):
    sigma = tf.matmul(sqrtSig, sqrtSig)
    term1 = tf.log(tf.matrix_determinant(sigma + 1e-8) + 1e-8)
    X_m = x - mean
    Shape = tf.shape(mean)
    reshapeX_m1 = tf.reshape(X_m, (Shape[0], Shape[1], 1, Shape[2]))
    reshapeX_m2 = tf.reshape(X_m, (Shape[0], Shape[1], Shape[2], 1))
    term2 = tf.matmul(tf.matmul(reshapeX_m1, tf.matrix_inverse(sigma + 1e-8)), reshapeX_m2)

    term1 = tf.reduce_mean(term1)
    term2 = tf.reduce_mean(term2)
    nll = 0.5 * (term1 + term2) / tf.cast(Shape[2], tf.float32) + 0.5*tf.log(2*np.pi)
    return nll
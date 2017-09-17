# Tools to build an RNN.
# Author: Yingru Liu
# Institute: Stony Brook University
import tensorflow as tf

# Build the hidden layers of the RNN
def inference_net(
        unitType,
        dimLayer,
        x
):
    numLayer = len(dimLayer) - 2
    hidden_out = x
    for i in range(numLayer):
        tf.name_scope('hidden_' + str(i + 1))
        if unitType == 'LSTM':
            lstm = tf.contrib.rnn.BasicLSTMCell(dimLayer + 1)
    pass
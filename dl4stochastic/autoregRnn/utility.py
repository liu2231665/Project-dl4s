# Tools to build an RNN.
# Author: Yingru Liu
# Institute: Stony Brook University
import tensorflow as tf

"""
Class: config - Basic configuration of the auto-regressive RNN.
"""
class config(object):
    unitType = 'LSTM'           # <string> the type of hidden units(LSTM/GRU/Tanh).
    dimLayer = []               # <scalar list> the size of each layers [input, hiddens, output].
    init_scale = 0.1            # <scalar> the initialized scales of the weight.
    learning_rate = 1.0         # <scalar> the learning rate
    max_epoch = 1               # <scalar> the maximum training epoches.
    max_steps = 100             # <scalar> the length of generative samples.
    batch_size = 20             # <scalar> the batch size.
    float = 'float32'           # <string> the type of float.
    Opt = 'SGD'                 # <string> the optimization method.
    savePath = None             # <string/None> the path to save the model.
    eventPath = None            # <string/None> the path to save the events for visualization.

#
"""
hidden_net: function to build the hidden layers of the RNN
input: x - network input indicated by <tensor placeholder>. 
       Config - configuration class.
output: cells - tensorflow symbol for the hidden layers of the multi-layer RNN.
        outputs.reshape - the output of last hidden layer.
        initializer - the initializer that may be used later.
"""
def hidden_net(
        x,
        Config=config(),
):
    # get the number of hidden layers.
    numLayer = len(Config.dimLayer) - 2
    # define the initializer.
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
    state = cells.zero_state(Config.batch_size, Config.float)
    #output: [batch_size, max_time, cell.output_size]
    outputs, _ = tf.nn.dynamic_rnn(cells, x, initial_state=state)

    return cells, outputs, initializer
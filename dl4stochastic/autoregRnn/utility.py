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
    max_steps=1                 # <scalar> the max length of sequence.
    batch_size = 1              # <scalar> the batch size
    init_scale = 0.1            # <scalar> the initialized scales of the weight.
    float = 'float32'           # <string> the type of float.
    Opt = 'SGD'                 # <string> the optimization method.
    savePath = None             # <string/None> the path to save the model.
    eventPath = None            # <string/None> the path to save the events for visualization.
    loadPath = None             # <string/None> the path to load the model.

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
        state = cells.zero_state(Config.batch_size, Config.float)

        #output: [batch_size, max_time, cell.output_size]
        outputs, _ = tf.nn.dynamic_rnn(cells, x, initial_state=state)
    return cells, outputs, initializer
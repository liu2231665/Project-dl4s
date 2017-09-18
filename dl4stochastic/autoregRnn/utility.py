# Tools to build an RNN.
# Author: Yingru Liu
# Institute: Stony Brook University
import tensorflow as tf

class config(object):
    unitType = 'LSTM'
    dimLayer = []
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    max_epoch = 1
    max_steps = 100
    batch_size = 20
    float = 'float32'

# Build the hidden layers of the RNN
def hidden_net(
        x,
        Config=config(),
):
    numLayer = len(Config.dimLayer) - 2

    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
    layers = []
    for i in range(numLayer):
        tf.variable_scope('hidden_' + str(i + 1), initializer=initializer)
        if Config.unitType == 'LSTM':
            layers.append(tf.nn.rnn_cell.LSTMCell(num_units=Config.dimLayer[i+1]))
        elif Config.unitType == 'GRU':
            layers.append(tf.nn.rnn_cell.GRUCell(num_units=Config.dimLayer[i + 1]))
        else:
            layers.append(tf.nn.rnn_cell.BasicRNNCell(num_units=Config.dimLayer[i + 1]))
    cells = tf.contrib.rnn.MultiRNNCell(layers, state_is_tuple=True)

    state = cells.zero_state(config.batch_size, config.float)
    outputs, _ = tf.nn.dynamic_rnn(cells, x, initial_state=state)

    return cells, tf.reshape(outputs, (-1, Config.dimLayer[-2])), initializer
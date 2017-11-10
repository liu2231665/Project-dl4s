"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: Tools to build an sequential VAE.
              ----2017.11.03
#########################################################################"""
import tensorflow as tf

"""#########################################################################
Function: buildHidden - build the hidden layers.
input: x - a placeholder that indicates the input data.
       dimLayer - dimension of each hidden layer.
       unitType - the type of recurrent units.
       init_scale - the initialization scaling of the weights.
output: cells - a tensorflow RNNcell object.
#########################################################################"""
def buildHidden(
        dimLayer,
        unitType,
        init_scale
):
    initializer = tf.random_uniform_initializer(-init_scale, init_scale)
    # <list> stacks of the hidden layers.
    layers = []
    for i in range(len(dimLayer)):
        tf.variable_scope('hidden_' + str(i + 1), initializer=initializer)
        if unitType == 'LSTM':
            layers.append(tf.nn.rnn_cell.LSTMCell(num_units=dimLayer[i]))
        elif unitType == 'GRU':
            layers.append(tf.nn.rnn_cell.GRUCell(num_units=dimLayer[i]))
        else:
            layers.append(tf.nn.rnn_cell.BasicRNNCell(num_units=dimLayer[i]))
    cells = tf.contrib.rnn.MultiRNNCell(layers, state_is_tuple=True)
    return cells

"""#########################################################################
Descriptions: Tools of the STORN.
              ----2017.11.03
#########################################################################"""
"""#########################################################################
Class: configSTORN - Basic configuration of the STORN models. 
       For the model details, please refer to:
       "Learning Stochastic Recurrent Networks" - arxiv.
        https://arxiv.org/abs/1411.7610 
#########################################################################"""
class configSTORN(object):
    """
    Elements outside the __init__ method are static elements.
    Elements inside the __init__ method are elements of the object.
    ----from Stackoverflow(https://stackoverflow.com/questions/9056957/correct-way-to-define-class-variables-in-python).
    """
    unitType = 'LSTM'           # <string> the type of hidden units(LSTM/GRU/Tanh).
    dimGen = []                 # <scalar list> the size of hidden layers in generating model.
    dimReg = []                 # <scalar list> the size of hidden layers in recognition model.
    dimInput = 100                # <scalar> the size of frame of the input.
    dimState = 100                # <scalar> the size of the stochastic layer.
    init_scale = 0.1            # <scalar> the initialized scales of the weight.
    float = 'float32'           # <string> the type of float.
    Opt = 'SGD'                 # <string> the optimization method.
    savePath = None             # <string/None> the path to save the model.
    eventPath = None            # <string/None> the path to save the events for visualization.
    loadPath = None             # <string/None> the path to load the model.

"""#########################################################################
Class: stornCell - Basic step of the STORN models. 
#########################################################################"""
class stornCell(tf.contrib.rnn.RNNCell):
    """
    __init__: the initialization function.
    input: configSTORN - configuration class in ./utility.
    output: None.
    """
    def __init__(self, configSTORN):
        self._dimState = configSTORN.dimState       # the dimension of stochastic layer
        self._dimGen = configSTORN.dimGen           # the dimension of each layer in generating model.
        self._dimReg = configSTORN.dimReg           # the dimension of each layer in recognition model.
        self._unitType = configSTORN.unitType       # the type of units for recurrent layers.
        self._init_scale = configSTORN.init_scale   # the initialized scale for the model.
        self.hiddenReg = buildHidden(self._dimReg, self._unitType, self._init_scale)    # the hidden layer part of the recognition model.
        self.hiddenGen = buildHidden(self._dimGen, self._unitType, self._init_scale)    # the hidden layer part of the generating model.

    @property
    def state_size(self):
        return self._dimState

    @property
    def output_size(self):
        # shape of mu, sig and generating hidden output.
        return (self._dimState, self._dimState, self._dimGen[-1])

    """
    __call__:
    input: x - the current input with size (batch, frame)
           state - the previous state of the cells.
           scope - indicate the variable scope.
    output: (muZ, sigZ, hg_t) - the mean and variance of P(Z_t|X_{1:t});
                                the generating hidden output that will be used by binSTORN/gaussSTORN.
            stateReg + stateGen - the new state of the cell.
    """
    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope('recognitionModel'):
                hr_t, stateReg = self.hiddenReg(x, state[0:len(self._dimReg)])     # recognition hidden output batch, frame)
                # define the output parameters of the recognizer outputs as [muZ, sigmaZ] of P(Z|X)
                Wz_mu = tf.get_variable('Wr_mu', shape=(self._dimReg[-1], self._dimState))
                bz_mu = tf.get_variable('br_mu', shape=self._dimState, initializer=tf.zeros_initializer)
                Wz_sig = tf.get_variable('Wr_sig', shape=(self._dimReg[-1], self._dimState))
                bz_sig = tf.get_variable('br_sig', shape=self._dimState, initializer=tf.zeros_initializer)
                # compute the [muZ, sigmaZ] of P(Z|X) with shape (batch, state)
                muZ = tf.matmul(hr_t, Wz_mu) + bz_mu
                sigZ = tf.nn.softplus(tf.matmul(hr_t, Wz_sig) + bz_sig) + 1e-8
                # generate the sample Z_t.
                # eps is the r.v of standard normal distribution with shape (batch, state)
                eps = tf.distributions.Normal(loc=0.0, scale=1.0
                                              ).sample(sample_shape=(tf.shape(x)[0], self._dimState))
                Z_t = muZ + sigZ * eps
            with tf.variable_scope('generateModel'):
                hg_t, stateGen = self.hiddenGen(tf.concat(axis=1, values=(x, Z_t)), state[len(self._dimReg):])      # generating hidden output batch, frame)

            return (muZ, sigZ, hg_t), stateReg + stateGen

    """
    zero_state: generate the zero initial state of the cells.
    input: batch_size - the batch size of data chunk.
           dtype - the data type.
    output: state0 + state1 - the initial zero states.
    """
    def zero_state(self, batch_size, dtype):
        state0 = self.hiddenReg.zero_state(batch_size, dtype)
        state1 = self.hiddenGen.zero_state(batch_size, dtype)
        return state0 + state1

"""#########################################################################
Class: halfStornCell - the recurrent cells of generating model of the STORN. 
#########################################################################"""
class halfStornCell(tf.contrib.rnn.RNNCell):
    """
    __init__: the initialization function.
    input: configSTORN - configuration class in ./utility.
    output: None.
    """
    def __init__(self, configSTORN, hiddenGen):
        self._dimState = configSTORN.dimState       # the dimension of stochastic layer
        self._dimGen = configSTORN.dimGen           # the dimension of each layer in generating model.
        self._unitType = configSTORN.unitType       # the type of units for recurrent layers.
        self._init_scale = configSTORN.init_scale   # the initialized scale for the model.
        self.hiddenGen = hiddenGen    # the hidden layer part of the generating model.

    @property
    def state_size(self):
        return self._dimState

    @property
    def output_size(self):
        # shape of mu, sig and generating hidden output.
        return self._dimGen[-1]

    """
    __call__:
    input: x - the current input with size (batch, frame)
           state - the previous state of the cells.
           scope - indicate the variable scope.
    output: (muZ, sigZ, hg_t) - the mean and variance of P(Z_t|X_{1:t});
                                the generating hidden output that will be used by binSTORN/gaussSTORN.
            stateReg + stateGen - the new state of the cell.
    """
    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            Z = tf.distributions.Normal(loc=0.0, scale=1.0).sample(sample_shape=(tf.shape(x)[0], self._dimState))
            with tf.variable_scope('generateModel'):
                hg_t, stateGen = self.hiddenGen(tf.concat(axis=1, values=(x, Z)), state)
                return hg_t, stateGen

    """
    zero_state: generate the zero initial state of the cells.
    input: batch_size - the batch size of data chunk.
           dtype - the data type.
    output: state - the initial zero states.
    """
    def zero_state(self, batch_size, dtype):
        state = self.hiddenGen.zero_state(batch_size, dtype)
        return state

"""#########################################################################
Function: buildTrainModel - build the model structure (Recognition + Genera
                            -ting) of the STORN.
input: x - a placeholder that indicates the input data. [batch, step, frame]
       graph - the default tf.graph that we build the model.
       hiddenGen - the hidden layer of the generating model.
       hiddenReg - the hidden layer of the recognition model.
       Config - model configuration.
output:
#########################################################################"""
def buildSTORN(
        x,
        graph,
        Config=configSTORN(),
):
    with graph.as_default():
        # define the variational cell of STORN
        allCell = stornCell(Config)
        #define the generating cell.
        halfCell = halfStornCell(Config, allCell.hiddenGen)

        # run the whole model.
        state = allCell.zero_state(tf.shape(x)[0], dtype=tf.float32)
        (muZ, sigZ, hg_t), _ = tf.nn.dynamic_rnn(allCell, x, initial_state=state)
        # run the generating model.
        state = halfCell.zero_state(tf.shape(x)[0], dtype=tf.float32)
        hiddenGen_t, _ = tf.nn.dynamic_rnn(halfCell, x, initial_state=state)
    return muZ[:, 0:-1, :], sigZ[:, 0:-1, :], hg_t[:, 0:-1, :], hiddenGen_t[:, 0:-1, :], \
           allCell, halfCell
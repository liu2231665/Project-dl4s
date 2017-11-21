"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: Tools to build an sequential VAE.
              ----2017.11.03
#########################################################################"""
import tensorflow as tf

"""#########################################################################
Function: buildRec - build the recurrent hidden layers.
input: x - a placeholder that indicates the input data.
       dimLayer - dimension of each hidden layer.
       unitType - the type of recurrent units.
       init_scale - the initialization scaling of the weights.
output: cells - a tensorflow RNNcell object.
#########################################################################"""
def buildRec(
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
Class: MLP - build the multilayer perceptron (MLP). 
input: init_scale - the initial scale.
       x - the input of network. (batch, time, frame)
       dimFor - the dimensions of layers in MLP.
#########################################################################"""
class MLP(object):
    def __init__(self, init_scale, dimInput, dimFor=[], unitType='relu'):
        self._dimInput = dimInput
        self._dimFor = dimFor
        self._dimOutput = self._dimInput if len(self._dimFor) == 0 else self._dimFor[-1]
        self._unitType = unitType
        self._init_scale = init_scale
        self._W = []
        self._b = []
        initializer = tf.random_uniform_initializer(-self._init_scale, self._init_scale)
        for l in range(len(self._dimFor)):
            if l == 0:
                self._W.append(tf.get_variable('W'+str(l), shape=(self._dimInput, self._dimFor[l])))
            else:
                self._W.append(tf.get_variable('W'+str(l), shape=(self._dimFor[l-1], self._dimFor[l])))
            self._b.append(tf.get_variable('b'+str(l), shape=self._dimFor[l], initializer=tf.zeros_initializer))

    def __call__(self, x):
        if len(self._dimFor) == 0:
            return x
        # build the network.
        xx = x
        for l in range(len(self._dimFor)):
                logit = tf.tensordot(xx, self._W[l], [[-1], [0]]) + self._b[l]
                if self._unitType == 'relu':
                    xx = tf.nn.relu(logit, name="relu-"+str(l))
                elif self._unitType == 'tanh':
                    xx = tf.nn.tanh(logit, name="tanh-"+str(l))
                elif self._unitType == 'sigmoid':
                    xx = tf.nn.sigmoid(logit, name="sigmoid-"+str(l))
                else:
                    raise ValueError("The unitType should be either relu, tanh or sigmoid!!")
        return xx



"""###############################################STORN####################################################"""
#####################################################
# Descriptions: Tools of the STORN.                 #
#             ----2017.11.11                        #
#####################################################

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
    ----from Stackoverflow(https://stackoverflow.com/questions/9056957/
    correct-way-to-define-class-variables-in-python).
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
        self.hiddenReg = buildRec(self._dimReg, self._unitType, self._init_scale)    # the hidden layer part of the recognition model.
        self.hiddenGen = buildRec(self._dimGen, self._unitType, self._init_scale)    # the hidden layer part of the generating model.

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

"""###############################################VRNN#####################################################"""
#####################################################
# Descriptions: Tools of the VRNN.                  #
#             ----2017.11.13                        #
#####################################################
"""#########################################################################
Class: configVRNN - Basic configuration of the VRNN models. 
       For the model details, please refer to:
       "A Recurrent Latent Variable Model for Sequential Data" - arxiv.
        https://arxiv.org/abs/1506.02216 
#########################################################################"""
class configVRNN(object):
    """
    Elements outside the __init__ method are static elements.
    Elements inside the __init__ method are elements of the object.
    ----from Stackoverflow(https://stackoverflow.com/questions/9056957/correct-way-to-define-class-variables-in-python).
    """
    recType = 'LSTM'            # <string> the type of recurrent hidden units(LSTM/GRU/Tanh).
    mlpType = 'relu'            # <string> the type of feedforward hidden units(relu/tanh/sigmoid).
    dimRec = []                 # <scalar list> the size of recurrent hidden layers.
    dimForX = []                # <scalar list> the size of feedforward hidden layers of input.
    dimForZ = []                # <scalar list> the size of feedforward hidden layers of stochastic layer.
    dimForEnc = []              # <scalar list> the size of feedforward hidden layers in the encoder.
    dimForDec = []              # <scalar list> the size of feedforward hidden layers in the decoder.
    dimInput = 100              # <scalar> the size of frame of the input.
    dimState = 100              # <scalar> the size of the stochastic layer.
    init_scale = 0.1            # <scalar> the initialized scales of the weight.
    float = 'float32'           # <string> the type of float.
    Opt = 'SGD'                 # <string> the optimization method.
    savePath = None             # <string/None> the path to save the model.
    eventPath = None            # <string/None> the path to save the events for visualization.
    loadPath = None             # <string/None> the path to load the model.

"""#########################################################################
Class: varCell - the variational cell of the VRNN models. 
#########################################################################"""
class varCell(tf.contrib.rnn.RNNCell):
    """
    __init__: the initialization function.
    input: configVRNN - configuration class in ./utility.
    output: None.
    """
    def __init__(self, config=configVRNN, train=True):
        self._dimState = config.dimState            # the dimension of stochastic layer
        self._dimInput = config.dimInput
        self._dimRec = config.dimRec                # the dimension of recurrent layers.
        self._dimMLPx = config.dimForX
        self._dimMLPz = config.dimForZ
        self._dimMLPenc = config.dimForEnc
        self._dimMLPdec = config.dimForDec
        self._recType = config.recType              # the type of units for recurrent layers.
        self._mlpType = config.mlpType              # the type of units for recurrent layers.
        self._init_scale = config.init_scale        # the initialized scale for the model.
        # the feedforward network of input X.
        with tf.variable_scope('mlpx'):
            self._mlpx = MLP(init_scale=self._init_scale, dimInput=self._dimInput, dimFor=self._dimMLPx)
        # the feedforward network of state Z.
        with tf.variable_scope('mlpz'):
            self._mlpz = MLP(init_scale=self._init_scale, dimInput=self._dimState, dimFor=self._dimMLPz)
        # the feedforward network for encoder.
        temp = self._dimRec[-1]
        if len(self._dimMLPx) != 0:
            temp += self._dimMLPx[-1]
        else:
            temp += self._dimInput

        with tf.variable_scope('mlpEnc'):
            self._mlpEnc = MLP(init_scale=self._init_scale, dimInput=temp, dimFor=self._dimMLPenc)
        # the feedforward network for decoder.
        temp = self._dimRec[-1]
        if len(self._dimMLPz) != 0:
            temp += self._dimMLPz[-1]
        else:
            temp += self._dimState

        with tf.variable_scope('mlpDec'):
            self._mlpDec = MLP(init_scale=self._init_scale, dimInput=temp, dimFor=self._dimMLPdec)
        # the recurrent network.
        self._rnn = buildRec(self._dimRec, self._recType, self._init_scale)
        #
        self._train = train
        #
        initializer = tf.random_uniform_initializer(-self._init_scale, self._init_scale)
        with tf.variable_scope('prior', initializer=initializer):
            self._Wp_mu = tf.get_variable('Wp_mu', shape=(self._dimRec[-1], self._dimState))
            self._bp_mu = tf.get_variable('bp_mu', shape=self._dimState, initializer=tf.zeros_initializer)
            self._Wp_sig = tf.get_variable('Wp_sig', shape=(self._dimRec[-1], self._dimState))
            self._bp_sig = tf.get_variable('bp_sig', shape=self._dimState, initializer=tf.zeros_initializer)
        with tf.variable_scope('encoder', initializer=initializer):
            if len(self._dimMLPenc) != 0:
                inputshape = self._dimMLPenc[-1]
            else:
                inputshape = self._dimRec[-1]
                if len(self._dimMLPx) != 0:
                    inputshape += self._dimMLPx[-1]
                else:
                    inputshape += self._dimInput
            self._Wenc_mu = tf.get_variable('Wenc_mu', shape=(inputshape, self._dimState))
            self._benc_mu = tf.get_variable('benc_mu', shape=self._dimState, initializer=tf.zeros_initializer)
            self._Wenc_sig = tf.get_variable('Wenc_sig', shape=(inputshape, self._dimState))
            self._benc_sig = tf.get_variable('benc_sig', shape=self._dimState, initializer=tf.zeros_initializer)

    @property
    def state_size(self):
        return self._dimState


    @property
    def output_size(self):
        if len(self._dimMLPdec) != 0:
            temp = self._dimMLPdec[-1]
        else:
            temp = self._dimRec[-1]
            if len(self._dimMLPz) != 0:
                temp += self._dimMLPz[-1]
            else:
                temp += self._dimState
        return (self._dimState, self._dimState, self._dimState, self._dimState,
                temp, self._dimRec[-1])

    """
    setGen: setting the generative models.
    """
    def setGen(self):
        self._train = False

    def setTrain(self):
        self._train = True

    """
    __call__:
    input: x - the current input with size (batch, frame)
           state - the previous state of the cells.
           scope - indicate the variable scope.
    output: 
    """
    def __call__(self, x, state, scope=None):
        if self._recType == 'LSTM':
            h_tm1 = state[-1][1]
        else:
            h_tm1 = state[-1]
        # Compute the prior.
        initializer = tf.random_uniform_initializer(-self._init_scale, self._init_scale)
        with tf.variable_scope('prior', initializer=initializer):
            # compute the mean and variance of P(Z) based on h_{t-1}
            prior_mu = tf.matmul(h_tm1, self._Wp_mu) + self._bp_mu
            prior_sig = tf.nn.softplus(tf.matmul(h_tm1, self._Wp_sig) + self._bp_sig) + 1e-8
        # Compute the encoder.
        with tf.variable_scope('encoder', initializer=initializer):
            xx = self._mlpx(x)
            hidden_enc = self._mlpEnc(tf.concat(axis=1, values=(xx, h_tm1)))
            # compute the mean and variance of the posterior P(Z|X).
            pos_mu = tf.matmul(hidden_enc, self._Wenc_mu) + self._benc_mu
            pos_sig = tf.nn.softplus(tf.matmul(hidden_enc, self._Wenc_sig) + self._benc_sig) + 1e-8

        # sample Z from the posterior.
        eps = tf.distributions.Normal(loc=0.0, scale=1.0
                                      ).sample(sample_shape=(tf.shape(x)[0], self._dimState))
        if self._train:
            z = pos_mu + pos_sig * eps
        else:
            z = prior_mu + prior_sig * eps
        # Compute the decoder.
        with tf.variable_scope('decoder', initializer=initializer):
            zz = self._mlpz(z)
            hidden_dec = self._mlpDec(tf.concat(axis=1, values=(zz, h_tm1)))
        # Unpdate the state.
        _, newState = self._rnn(tf.concat(axis=1, values=(xx, zz)), state)
        return (prior_mu, prior_sig, pos_mu, pos_sig, hidden_dec, h_tm1), newState

    """
    zero_state: generate the zero initial state of the cells.
    input: batch_size - the batch size of data chunk.
           dtype - the data type.
    output: state0 - the initial zero states.
    """
    def zero_state(self, batch_size, dtype):
        state0 = self._rnn.zero_state(batch_size, dtype)
        return state0

"""#########################################################################
Function: buildVRNN - build the whole graph of VRNN. 
input: x - a placeholder that indicates the input data. [batch, step, frame]
       graph - the default tf.graph that we build the model.
       Config - model configuration.
#########################################################################"""
def buildVRNN(
        x,
        graph,
        Config=configVRNN(),
):
    with graph.as_default():
        # define the variational cell of VRNN
        allCell = varCell(Config)

        # run the whole model.
        state = allCell.zero_state(tf.shape(x)[0], dtype=tf.float32)
        (prior_mu, prior_sig, pos_mu, pos_sig, hidden_dec, h_tm1), _ = tf.nn.dynamic_rnn(allCell, x, initial_state=state)
        return prior_mu, prior_sig, pos_mu, pos_sig, hidden_dec, h_tm1, allCell


"""###############################################SRNN#####################################################"""
#####################################################
# Descriptions: Tools of the SRNN.                  #
#             ----2017.11.15                        #
#####################################################
"""#########################################################################
Class: configSRNN - Basic configuration of the SRNN models. 
       For the model details, please refer to:
       "Sequential Neural Models with Stochastic Layers" - arxiv.
        https://arxiv.org/abs/1605.07571
#########################################################################"""
class configSRNN(object):
    recType = 'LSTM'            # <string> the type of recurrent hidden units(LSTM/GRU/Tanh).
    mlpType = 'relu'            # <string> the type of feedforward hidden units(relu/tanh/sigmoid).
    mode = 'smooth'             # <string> indicate the operating mode of SRNN (smooth/filter).
    dimRecD = []                # <scalar list> the size of forward recurrent hidden layers.
    dimRecA = []                # <scalar list> the size of backward recurrent hidden layers.
    dimEnc = []
    dimDec = []
    dimMLPx = []                # <scalar list> the size of MLP of X.
    dimInput = 100              # <scalar> the size of frame of the input.
    dimState = 100              # <scalar> the size of the stochastic layer.
    init_scale = 0.1            # <scalar> the initialized scales of the weight.
    float = 'float32'           # <string> the type of float.
    Opt = 'SGD'                 # <string> the optimization method.
    savePath = None             # <string/None> the path to save the model.
    eventPath = None            # <string/None> the path to save the events for visualization.
    loadPath = None             # <string/None> the path to load the model.

"""#########################################################################
Class: stoCell - the stochastic cell of the SRNN models. 
#########################################################################"""
class stoCell(tf.contrib.rnn.RNNCell):
    """
    __init__: the initialization function.
    input: configSRNN - configuration class in ./utility.
           train - indicate whether the model is trained or sampling.
    output: None.
    """
    def __init__(self, config=configSRNN, train=True):
        self._train = train
        self._dimState = config.dimState
        self._dimInput = config.dimInput
        self._dimEnc = config.dimEnc
        self._dimDec = config.dimDec
        self._dimDt = config.dimRecD[-1]
        if len(config.dimRecA) != 0:
            self._dimAt = config.dimRecA[-1]
        else:
            self._dimAt = config.dimRecD[-1] + config.dimInput
        # define the hidden output shape of the encoder.
        if len(self._dimEnc) != 0:
            self._dimOutEnc = self._dimEnc[-1]
        else:
            self._dimOutEnc = self._dimAt + self._dimState
        #
        self._recType = config.recType              # the type of units for recurrent layers.
        self._mlpType = config.mlpType              # the type of units for recurrent layers.
        self._init_scale = configSRNN.init_scale    # the initialized scale for the model.
        # Decoder Input = [Z{t}, d{t}]
        with tf.variable_scope('DecMLP'):
            self._decoder = MLP(self._init_scale, self._dimState+self._dimDt, self._dimDec, self._mlpType)

        # Encoder Input = [Z{t-1}, a{t}]
        with tf.variable_scope('EncMLP'):
            self._encoder = MLP(self._init_scale, self._dimState+self._dimAt, self._dimEnc, self._mlpType)
        # system's parameter for the prior P(Z)= NN(Z{t-1}, d{t})
        initializer = tf.random_uniform_initializer(-self._init_scale, self._init_scale)
        with tf.variable_scope('prior', initializer=initializer):
            self._Wp_mu = tf.get_variable('Wp_mu', shape=(self._dimState+self._dimDt, self._dimState))
            self._bp_mu = tf.get_variable('bp_mu', shape=self._dimState, initializer=tf.zeros_initializer)
            self._Wp_sig = tf.get_variable('Wp_sig', shape=(self._dimState+self._dimDt, self._dimState))
            self._bp_sig = tf.get_variable('bp_sig', shape=self._dimState, initializer=tf.zeros_initializer)

        with tf.variable_scope('encoder', initializer=initializer):
            self._Wpos_mu = tf.get_variable('Wpos_mu', shape=(self._dimOutEnc, self._dimState))
            self._bpos_mu = tf.get_variable('bpos_mu', shape=self._dimState, initializer=tf.zeros_initializer)
            self._Wpos_sig = tf.get_variable('Wpos_sig', shape=(self._dimOutEnc, self._dimState))
            self._bpos_sig = tf.get_variable('bpos_sig', shape=self._dimState, initializer=tf.zeros_initializer)

    """
    __call__:
    input: x - the current input with size (batch, frame) where frame = [d_t, a_t]
               the bounds between them is 0~dimDt, dimDt~end
           state - the previous state of the cells.
           scope - indicate the variable scope.
    output: 
    """
    def __call__(self, x, state, scope=None):
        with tf.variable_scope('prior'):
            # compute the mean and std of P(Z) based on [Z{t-1}, d_{t-1}]
            d_tm1 = x[:, 0:self._dimDt]
            prior_mu = tf.matmul(tf.concat(axis=1, values=(state, d_tm1)), self._Wp_mu) + self._bp_mu
            prior_sig = tf.nn.softplus(tf.matmul(tf.concat(axis=1, values=(state, d_tm1))
                                                 , self._Wp_sig) + self._bp_sig) + 1e-8
        with tf.variable_scope('encoder'):
            # build post mean and std of the inference network by NN(Z{t-1}, at)
            a_t = x[:, self._dimDt:]
            actPos = self._encoder(tf.concat(axis=1, values=(state, a_t)))
            pos_mu = tf.matmul(actPos, self._Wpos_mu) + self._bpos_mu
            pos_sig = tf.nn.softplus(tf.matmul(actPos, self._Wpos_sig) + self._bpos_sig) + 1e-8
            # sample Z/NewSate from the posterior.
            eps = tf.distributions.Normal(loc=0.0, scale=1.0
                                          ).sample(sample_shape=(tf.shape(x)[0], self._dimState))
            if self._train:
                z = pos_mu + pos_sig * eps
            else:
                z = prior_mu + prior_sig * eps
        with tf.variable_scope('decoder'):
            # Compute the decoder with input = [Z{t}, d{t}]
            hidden_dec = self._decoder(tf.concat(axis=1, values=(z, d_tm1)))

        return (prior_mu, prior_sig, pos_mu, pos_sig, hidden_dec), z

    """
    zero_state: generate the zero initial state of the cells.
    input: batch_size - the batch size of data chunk.
           dtype - the data type.
    output: state0 - the initial zero states.
    """
    def zero_state(self, batch_size, dtype):
        state0 = tf.distributions.Normal(loc=0.0, scale=1.0).sample(sample_shape=(batch_size, self._dimState))
        return state0

    @property
    def state_size(self):
        return self._dimState

    @property
    def output_size(self):
        if len(self._dimDec) != 0:
            temp = self._dimDec[-1]
        else:
            temp = self._dimState + self._dimDt
        return (self._dimState, self._dimState, self._dimState, self._dimState,
                temp)


"""#########################################################################
Function: buildSRNN - build the whole graph of SRNN. 
input: x - a placeholder that indicates the input data. [batch, step, frame]
       graph - the default tf.graph that we build the model.
       Config - model configuration.
#########################################################################"""
def buildSRNN(
        x,
        graph,
        Config=configSRNN(),
):
    with graph.as_default():
        # define the variational cell of VRNN
        MLPx = MLP(Config.init_scale, Config.dimInput, Config.dimMLPx, Config.mlpType)
        with tf.variable_scope("forwardCell"):
            forwardCell = buildRec(Config.dimRecD, Config.recType, Config.init_scale)  # the hidden layer part of the recognition model.
            # run the forward recurrent layers to compute the deterministic transition.
            paddings = tf.constant([[0, 0], [1, 0], [0, 0]])
            xx = tf.pad(x[:, 0:-1, :], paddings)
            state = forwardCell.zero_state(tf.shape(xx)[0], dtype=tf.float32)
            d_t, _ = tf.nn.dynamic_rnn(forwardCell, MLPx(xx), initial_state=state)
        # run the backward recurrent layers or MLP to compute a_t.
        with tf.variable_scope("backward"):
            if Config.mode == 'smooth':
                backwardCell = buildRec(Config.dimRecA, Config.recType, Config.init_scale)
                state = backwardCell.zero_state(tf.shape(x)[0], dtype=tf.float32)
                a_t, _ = tf.nn.dynamic_rnn(backwardCell, tf.reverse(tf.concat(axis=-1, values=(d_t, x)), [1]), initial_state=state)
            elif Config.mode == 'filter':
                backwardCell = MLP(Config.init_scale, Config.dimRecD[-1] + Config.dimInput,
                                   Config.dimRecA, Config.mlpType)
                a_t = backwardCell(tf.concat(axis=-1, values=(d_t, x)))
            else:
                a_t = None
                raise ValueError("The operating mode is not correct!!(Should be smooth/filter)")
        # the state space model cell.
        SSM = stoCell(Config)
        state = SSM.zero_state(tf.shape(x)[0], dtype=tf.float32)
        (prior_mu, prior_sig, pos_mu, pos_sig, hidden_dec), _ = tf.nn.dynamic_rnn(SSM, tf.concat(axis=-1, values=(d_t, a_t)), initial_state=state)
        return prior_mu, prior_sig, pos_mu, pos_sig, hidden_dec
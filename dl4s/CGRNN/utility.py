"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: Tools to build an Chain Graph RNN.
              ----2017.12.18
#########################################################################"""
import tensorflow as tf
from dl4s.SeqVAE.utility import buildRec, MLP
from dl4s.TRBM.RBM import bin_ssRBM, mu_ssRBM

"""#########################################################################
Class: configCGRNN - Basic setting of the CGRNN models. 
#########################################################################"""
class configCGRNN:
    aisRun = 100        # <scalar> the number of samples of AIS.
    aisLevel = 10       # <scalar> the number of intermediate proposal distributions of AIS.
    Gibbs = 15          # <scalar> the steps of Gibbs sampling.
    recType = 'LSTM'
    mlpType = 'relu'
    dimMlp = []
    dimRec = []
    mode = 'full'       # <string> indicate the mode of feedback. (D/S/full)
    dimInput = 100      # <scalar> the size of frame of the input.
    dimState = 100      # <scalar> the size of the stochastic layer.
    init_scale = 0.01   # <scalar> the initialized scales of the weight.
    float = 'float32'   # <string> the type of float.
    Opt = 'SGD'         # <string> the optimization method.
    savePath = None     # <string/None> the path to save the model.
    eventPath = None    # <string/None> the path to save the events for visualization.
    loadPath = None     # <string/None> the path to load the model.
    W_Norm = True
    alphaTrain = True
    muTrain = True
    phiTrain = True

"""#########################################################################
Class: CGCell - Basic step of the CGRNN models. 
#########################################################################"""
class CGCell(tf.contrib.rnn.RNNCell):
    """
    __init__: the initialization function.
    input: configSTORN - configuration class in ./utility.
    output: None.
    """
    def __init__(self, config, inputType='binary'):
        self._config = config
        self._mode = config.mode
        self._gibbs = config.Gibbs
        self._dimRec = config.dimRec
        self._dimInput = config.dimInput
        self._dimState = config.dimState
        self._inputType = inputType            # the dimension of stochastic layer
        self._init_scale = config.init_scale   # the initialized scale for the model.
        with tf.variable_scope('CGcell'):
            self.mlp = MLP(config.init_scale, config.dimInput, config.dimMlp, config.mlpType)
            self.rnnCell = buildRec(config.dimRec, config.recType,
                                    config.init_scale)    # the hidden layer part of the recognition model.
            # system's parameter for the prior P(Z)= NN(Z{t-1}, d{t})
            initializer = tf.random_uniform_initializer(-self._init_scale, self._init_scale)
            dim = self._dimRec[-1]
            with tf.variable_scope('feedback', initializer=initializer):
                #
                self._W_bv = tf.get_variable('W_bv', shape=(dim, self._dimInput))
                self._b_bv = tf.get_variable('b_bv', shape=self._dimInput, initializer=tf.zeros_initializer)
                self._W_bh = tf.get_variable('W_bh', shape=(dim, self._dimState))
                self._b_bh = tf.get_variable('b_bh', shape=self._dimState, initializer=tf.zeros_initializer)
                if self._inputType == 'continuous':
                    self._W_gamma = tf.get_variable('W_gamma', shape=(dim, self._dimInput))
                    self._b_gamma = tf.get_variable('b_gamma', shape=self._dimInput,
                                                    initializer=tf.zeros_initializer)
            # create the RBM component.
            with tf.variable_scope('RBM', initializer=initializer):
                if self._inputType == 'binary':
                    self.RBM = bin_ssRBM(dimV=self._config.dimInput, dimH=self._config.dimState,
                                         init_scale=self._init_scale,alphaTrain=self._config.alphaTrain,
                                         muTrain=self._config.muTrain, bv=self._b_bv, bh=self._b_bh,
                                         k=self._gibbs, CGRNN=True)
                elif self._inputType == 'continuous':
                    self.RBM = mu_ssRBM(dimV=self._config.dimInput, dimH=self._config.dimState,
                                        init_scale=self._config.init_scale,
                                        alphaTrain=self._config.alphaTrain,
                                        muTrain=self._config.muTrain, phiTrain=self._config.phiTrain,
                                        k=self._gibbs, CGRNN=True)
                else:
                    raise ValueError("The input type should be either binary or continuous!!")


    """
    __call__:
    input: x - the current input with size (batch, frame).
           state - the previous state of the cells. [newH, hidden, rnnstate]
           scope - indicate the variable scope.
           generative - indicate whether the cell is used for generating samples.
    output: 
    """
    def __call__(self, x, state, scope=None, generative=False, gibbs=None):
        k = gibbs if gibbs is not None else self._gibbs
        with tf.variable_scope('CGcell'):
            hidden = state[0]
            rnnstate = state[1:]
            if self._inputType == 'binary':
                bvt = tf.tensordot(hidden, self._W_bv, [[-1], [0]]) + self._b_bv
                bht = tf.tensordot(hidden, self._W_bh, [[-1], [0]]) + self._b_bh
                # run the RBM to generate necessary variable.
                newV, newH, newS, muV, muH, muS = self.RBM(xt=x, bvt=bvt, bht=bht, k=k)
                xx = newV if generative else x
                if self._mode == 'D':
                    hidden, newState = self.rnnCell(self.mlp(xx), rnnstate)
                elif self._mode == 'S':
                    hidden, newState = self.rnnCell(newS*newH, rnnstate)
                else:
                    hidden, newState = self.rnnCell(tf.concat([self.mlp(xx), newS*newH], axis=1), rnnstate)
                return (newV, newH, newS, muV, muH, muS, bvt, bht), \
                       (hidden,) + newState

            elif self._inputType == 'continuous':
                bvt = tf.tensordot(hidden, self._W_bv, [[-1], [0]]) + self._b_bv
                bht = tf.tensordot(hidden, self._W_bh, [[-1], [0]]) + self._b_bh
                gammat = tf.nn.softplus(tf.tensordot(
                    hidden, self._W_gamma, [[-1], [0]]) + self._b_gamma)
                # run the RBM to generate necessary variable.
                newV, newH, newS, muV, muH, muS = self.RBM(xt=x, bvt=bvt, bht=bht, gammat=gammat, k=k)
                xx = newV if generative else x
                if self._mode == 'D':
                    hidden, newState = self.rnnCell(self.mlp(xx), rnnstate)
                elif self._mode == 'S':
                    hidden, newState = self.rnnCell(newS*newH, rnnstate)
                else:
                    hidden, newState = self.rnnCell(tf.concat([self.mlp(xx), newS*newH], axis=1), rnnstate)
                return (newV, newH, newS, muV, muH, muS, bvt, bht, gammat), \
                       (hidden,) + newState
            else:
                raise ValueError("The input type should be either binary or continuous!!")

    """
    zero_state: generate the zero initial state of the cells.
    input: batch_size - the batch size of data chunk.
           dtype - the data type.
    output: state0 - the initial zero states.
    """
    def zero_state(self, batch_size, dtype):
        state0 = self.rnnCell.zero_state(batch_size, dtype)
        #H0 = tf.zeros(shape=(batch_size, self._dimState))
        hidden0 = tf.zeros(shape=(batch_size, self._dimRec[-1]))
        return (hidden0,) + state0

    @property
    def state_size(self):
        return self._dimState

    @property
    def output_size(self):
        if self._inputType == 'binary':
            return (self._dimInput, self._dimState, self._dimState,
                    self._dimInput, self._dimState, self._dimState,
                    self._dimInput, self._dimState)
        elif self._inputType == 'continuous':
            return (self._dimInput, self._dimState, self._dimState,
                    self._dimInput, self._dimState, self._dimState,
                    self._dimInput, self._dimState, self._dimInput)

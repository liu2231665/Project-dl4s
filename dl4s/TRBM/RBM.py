# TODO:
"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the file contains the model description of RBM.
              ----2017.11.19
Notes: in our packages, we use the NVIL to define a lower bound of the log
       likelihood of the RBM. For more details, please refer to
       "Neural Variational Inference and Learning in Undirected Graphical Models"
       <http://web.stanford.edu/~kuleshov/papers/nips2017.pdf>
#########################################################################"""

import tensorflow as tf
import numpy as np
from dl4s.SeqVAE.utility import MLP

"""#########################################################################
Class: _RBM - the basic class of a Restricted Boltzmann Machine.
              for model details, please refer to
              "A Practical Guide to Training Restricted Boltzmann Machines"
              <https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf>
#########################################################################"""
class _RBM(object):
    """#########################################################################
    __init__:the initialization function.
    input: dimV - the frame dimension of the input vector.
           dimH - the frame dimension of the latent state.
           init_scale - the initial scale of system parameters.
           x - the input tensor that maybe provided by outer parts. None in
               default and the RBM will define its own input tensor.
           W - the weight matrix that maybe provided by outer parts. None in
               default and the RBM will define its own input tensor.
           bv - the visible bias that maybe provided by outer parts. None in
               default and the RBM will define its own visible bias.
           bh - the hidden bias that maybe provided by outer parts. None in
               default and the RBM will define its own hidden bias.
           scope - using to define the variable_scope.
           loadPath - the path to load the saved model.
    output: None.
    #########################################################################"""
    def __init__(
            self,
            dimV,
            dimH,
            init_scale,
            x=None,
            W=None,             # W.shape = [dimV, dimH]
            bv=None,
            bh=None,
            scope=None,
            loadPath=None,
            Opt='SGD'
    ):
        # <Tensorflow Session>.
        self._sess = tf.Session()
        # the proposal distribution.
        self._Q = None
        # Check whether the configuration is correct.
        if x is not None and x.shape[-1] != dimV:
            raise ValueError("You have provided a input tensor but the last shape is not equal to [dimV]!!")
        if W is not None and W.shape[0] != dimV and W.shape[1] !=dimH:
            raise ValueError("You have provided a W tensor but the shape is not equal to [dimV, dimH]!!")
        if bv is not None and bv.shape[-1] != dimV:
            raise ValueError("You have provided a bv tensor but the last shape is not equal to [dimV]!!")
        if bh is not None and bh.shape[-1] != dimH:
            raise ValueError("You have provided a bh tensor but the last shape is not equal to [dimH]!!")

        self._dimV = dimV
        self._dimH = dimH
        self._loadPath = loadPath
        if scope is None:
            self._scopeName = 'RBM-' + str(np.random.randn())
        else:
            self._scopeName = scope

        # create the system parameters.
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.variable_scope(self._scopeName, initializer=initializer):
            #
            self._W = W if W is not None else tf.get_variable('W', shape=[dimV, dimH])
            self._bv = bv if bv is not None else tf.get_variable('bv', shape=dimV, initializer=tf.zeros_initializer)
            self._bh = bv if bv is not None else tf.get_variable('bh', shape=dimH, initializer=tf.zeros_initializer)
            # if the RBM component is used to build sequential models like RNN-RBM, the input x should be provided as
            # x = [batch, frame]. O.w, we define it as non-temporal data with shape [batch,frame].
            self._V = x if x is not None else tf.placeholder(dtype=tf.float32, shape=[None, dimV], name='V')
            # <tensor placeholder> learning rate.
            self.lr = tf.placeholder(dtype='float32', shape=(), name='learningRate')
            # the loss per bit.
            self._loss = None
            # the updating step.
            self._train_step = None
            # <Tensorflow Optimizer>.
            if Opt == 'Adadelta':
                self._optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
            elif Opt == 'Adam':
                self._optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            elif Opt == 'Momentum':
                self._optimizer = tf.train.MomentumOptimizer(self.lr, 0.9)
            elif Opt == 'SGD':
                self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            else:
                raise (ValueError("Config.Opt should be either 'Adadelta', 'Adam', 'Momentum' or 'SGD'!"))


    """#########################################################################
    sampleHgivenV: the inference direction of the RBM.
    input: V - the input vector, could be [batch, dimV].
           W - the weight matrix provided by outer network. Using default one if
               None.
           bh - the hidden bias provided by outer network. Using default one if
                None.
    output: newH - the new binary latent states sampled from P(H|V).
            Ph_v - the Bernoulli distribution P(H|V), which is also mean(H|V).
    #########################################################################"""
    def sampleHgivenV(self, V, W=None, bh=None):
        Wt = W if W is not None else self._W
        bht = bh if bh is not None else self._bh
        # shape of tensordot = [batch, dimH]
        Ph_v = tf.nn.sigmoid(tf.tensordot(V, Wt, [[-1], [0]]) + bht)
        newH = tf.distributions.Bernoulli(probs=Ph_v, dtype=tf.float32).sample()
        return newH, Ph_v

    """#########################################################################
    sampleVgivenH: the generative direction of the RBM.
    input: H - the latent state, could be [batch, dimH].
           W - the weight matrix provided by outer network. Using default one if
               None.
           bv - the visible bias provided by outer network. Using default one if
                None.
    output: new sample and probability depends on the observation form.
    #########################################################################"""
    def sampleVgivenH(self, H, W=None, bv=None):
        return None, None

    """#########################################################################
    GibbsSampling: Gibbs sampling.
    input: V - the input vector, could be [batch, dimV].
           W - the weight matrix provided by outer network. Using default one if
               None.
           bv - the visible bias provided by outer network. Using default one if
                None.
           bh - the hidden bias provided by outer network. Using default one if
                None.
           k - the running times of the Gibbs sampling. 1 in default.
    output: newV - new sample of V.
            newH - new sample of H.
            Ph_v - P(H|V).
            Pv_h - P(V|H).
    #########################################################################"""
    def GibbsSampling(self, V, W=None, bv=None, bh=None, k=1):
        newV = V
        newH = None
        Ph_v = None
        Pv_h = None
        for i in range(k):
            newH, Ph_v = self.sampleHgivenV(newV, W, bh)
            newV, Pv_h = self.sampleVgivenH(newH, W, bv)
            if newV is None:
                raise ValueError("You have not yet define the sampleVgivenH!!")
        return newV, newH, Ph_v, Pv_h

    """#########################################################################
    train_function: compute the loss and update the tensor variables.
    input: input - numerical input with shape [batch, dimV].
           lrate - <scalar> learning rate.
    output: the loss value.
    #########################################################################"""
    def train_function(self, input, lrate):
        _, loss_value = self._sess.run([self._train_step, self._loss],
                                       feed_dict={self._V: input, self.lr: lrate})
        return loss_value

    """#########################################################################
    FreeEnergy: the free energy function.
    input: V - the latent state, could be [batch, dimV].
           W - the weight matrix provided by outer network. Using default one if
               None.
           bv - the visible bias provided by outer network. Using default one if
                None.
           bh - the hidden bias provided by outer network. Using default one if
                None.
    output: the average free energy per frame with shape [batch]
    #########################################################################"""
    def FreeEnergy(self, V, W=None, bv=None, bh=None):
        return

    """#########################################################################
    ComputeLoss: define the loss of the log-likelihood with given proposal
                Q. If we are going to introduce complicated model like VAE to be
                the proposal. We should define the lower bound outer the class.
    input: V - the latent state, could be [batch, dimV].
           samplesteps - the number of sample to be drawn. Default is 1.
           W - the weight matrix provided by outer network. Using default one if
               None.
           bv - the visible bias provided by outer network. Using default one if
                None.
           bh - the hidden bias provided by outer network. Using default one if
                None.
    output: the average loss per frame in tensor.
    #########################################################################"""
    def ComputeLoss(self, V, samplesteps=10, W=None, bv=None, bh=None):
        Wt = W if W is not None else self._W
        # the shape of bvt could be [dimV].
        bvt = bv if bv is not None else self._bv
        # the shape of bvt could be [dimH].
        bht = bv if bh is not None else self._bh
        # samples.shape = [batch, dimV].
        samples, _, _, _ = self.GibbsSampling(V=V, k=samplesteps)
        # negative phase with shape [samples].
        negPhase = tf.reduce_mean(self.FreeEnergy(samples, Wt, bvt, bht))
        # positive phase with shape [batch].
        posPhase = tf.reduce_mean(self.FreeEnergy(V, Wt, bvt, bht))
        return posPhase - negPhase

    """#########################################################################
    _runSession: initialize the graph or restore from the load path.
    input: None.
    output: None.
    #########################################################################"""
    def _runSession(self):
        if self._loadPath is None:
            self._sess.run(tf.global_variables_initializer())
        else:
            saver = tf.train.Saver()
            saver.restore(self._sess, self._loadPath)
        return

    """#########################################################################
    saveModel:save the trained model into disk.
    input: savePath - another saving path, if not provide, use the default path.
    output: None
    #########################################################################"""
    def saveModel(self, savePath=None):
        # Create a saver.
        saver = tf.train.Saver()
        saver.save(self._sess, savePath)
        return

    """#########################################################################
    loadModel:load the model from disk.
    input: loadPath - another loading path, if not provide, use the default path.
    output: None
    #########################################################################"""
    def loadModel(self, loadPath=None):
        # Create a saver.
        saver = tf.train.Saver()
        if loadPath is None:
            if self._loadPath is not None:
                saver.restore(self._sess, self._loadPath)
            else:
                raise (ValueError("No loadPath is given!"))
        else:
            saver.restore(self._sess, loadPath)
        return



"""#########################################################################
Class: binRBM - the RBM model for binary observations
#########################################################################"""
class binRBM(_RBM, object):
    """#########################################################################
    __init__:the initialization function.
    input: dimV - the frame dimension of the input vector.
           dimH - the frame dimension of the latent state.
           init_scale - the initial scale of system parameters.
           x - the input tensor that maybe provided by outer parts. None in
               default and the RBM will define its own input tensor.
           Q - the tensor that represents the proposal distribution Q for NVIL.
               None in default and dimQ should be provide to build the Q.
           dimQ - the number of kernal in Q.
           W - the weight matrix that maybe provided by outer parts. None in
               default and the RBM will define its own input tensor.
           bv - the visible bias that maybe provided by outer parts. None in
               default and the RBM will define its own visible bias.
           bh - the hidden bias that maybe provided by outer parts. None in
               default and the RBM will define its own hidden bias.
           scope - using to define the variable_scope.
    output: None.
    #########################################################################"""
    def __init__(
            self,
            dimV,
            dimH,
            init_scale,
            x=None,
            W=None,  # W.shape = [dimV, dimH]
            bv=None,
            bh=None,
            scope=None
    ):
        super().__init__(dimV, dimH, init_scale, x, W, bv, bh, scope)
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        # Define the proposal Q.
        with tf.variable_scope(self._scopeName, initializer=initializer):
            # the training loss is per frame.
            self._loss = self.ComputeLoss(V=self._V)
            self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self._train_step = self._optimizer.minimize(self._loss)
            self._runSession()

    """#########################################################################
    sampleVgivenH: the generative direction of the RBM.
    input: H - the latent state, could be [batch, dimH].
           W - the weight matrix provided by outer network. Using default one if
               None.
           bv - the visible bias provided by outer network. Using default one if
                None.
    output: newV - the new sample of the visible units.
            Pv_h - P(V|H).
    #########################################################################"""
    def sampleVgivenH(self, H, W=None, bv=None):
        Wt = W if W is not None else self._W
        bvt = bv if bv is not None else self._bv
        # shape of tensordot = [batch, dimH]
        Pv_h = tf.nn.sigmoid(tf.tensordot(H, tf.transpose(Wt), [[-1], [0]]) + bvt)
        newV = tf.distributions.Bernoulli(probs=Pv_h, dtype=tf.float32).sample()
        return newV, Pv_h

    """#########################################################################
    FreeEnergy: the free energy function.
    input: V - the latent state, could be [batch, dimV].
           W - the weight matrix provided by outer network. Using default one if
               None.
           bv - the visible bias provided by outer network. Using default one if
                None.
           bh - the hidden bias provided by outer network. Using default one if
                None.
    output: the average free energy per frame with shape [batch]
    #########################################################################"""
    def FreeEnergy(self, V, W=None, bv=None, bh=None):
        Wt = W if W is not None else self._W
        # the shape of bvt could be [dimV].
        bvt = bv if bv is not None else self._bv
        # the shape of bvt could be [dimH].
        bht = bh if bh is not None else self._bh
        #
        term1 = V * bvt
        term2 = tf.nn.softplus(tf.tensordot(V, Wt, [[-1], [0]]) + bht)
        return -tf.reduce_sum(term1, axis=-1) - tf.reduce_sum(term2, axis=-1)

    """#########################################################################
    ComputeLoss: define the loss of the log-likelihood with given proposal
                Q. If we are going to introduce complicated model like VAE to be
                the proposal. We should define the lower bound outer the class.
    input: V - the latent state, could be [batch, dimV].
           samplesteps - the number of sample to be drawn. Default is 1.
           W - the weight matrix provided by outer network. Using default one if
               None.
           bv - the visible bias provided by outer network. Using default one if
                None.
           bh - the hidden bias provided by outer network. Using default one if
                None.
    output: the average loss per frame in tensor.
    #########################################################################"""
    def AIS(self, run=100, steps=100, W=None, bv=None, bh=None):
        Wt = W if W is not None else self._W
        # the shape of bvt could be [dimV].
        bvt = bv if bv is not None else self._bv
        # the shape of bvt could be [dimH].
        bht = bv if bh is not None else self._bh
        pass





"""#########################################################################
Class: gaussRBM - the RBM model for continuous-value observations
#########################################################################"""
class gaussRBM(_RBM, object):
    pass

"""#########################################################################
Class: _ssRBM - the spike and slab Restricted Boltzmann Machine.
#########################################################################"""
class _ssRBM(object):
    pass
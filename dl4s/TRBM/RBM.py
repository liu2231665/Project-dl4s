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
            scope=None
    ):
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
        if scope is None:
            self._scopeName = 'RBM-' + str(np.random.randn())
        else:
            self._scopeName = scope

        # create the system parameters.
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.variable_scope(self._scopeName, initializer=initializer):
            self._W = W if W is not None else tf.get_variable('W', shape=[dimV, dimH])
            self._bv = bv if bv is not None else tf.get_variable('bv', shape=dimV, initializer=tf.zeros_initializer)
            self._bh = bv if bv is not None else tf.get_variable('bh', shape=dimH, initializer=tf.zeros_initializer)
            # if the RBM component is used to build sequential models like RNN-RBM, the input x should be provided as
            # x = [batch, steps, frame]. O.w, we define it as non-temporal data with shape [batch,frame].
            self._V = x if x is not None else tf.placeholder(dtype=tf.float32, shape=[None, dimV], name='V')
            # <tensor placeholder> learning rate.
            self.lr = tf.placeholder(dtype='float32', shape=(), name='learningRate')

    """#########################################################################
    sampleHgivenV: the inference direction of the RBM.
    input: V - the input vector, could be [batch, steps, frame]/[batch, frame].
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
        # shape of tensordot = [batch, steps, dimH]/[batch, dimH]
        Ph_v = tf.nn.sigmoid(tf.tensordot(V, Wt, [[-1], [0]]) + bht)
        newH = tf.distributions.Bernoulli(probs=Ph_v, dtype=tf.float32).sample()
        return newH, Ph_v

    """#########################################################################
    sampleVgivenH: the generative direction of the RBM.
    input: H - the latent state, could be [batch, steps, frame]/[batch, frame].
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
    input: V - the input vector, could be [batch, steps, frame]/[batch, frame].
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
            Q=None,
            dimQ=1,
            W=None,  # W.shape = [dimV, dimH]
            bv=None,
            bh=None,
            scope=None
    ):
        super().__init__(dimV, dimH, init_scale, x, W, bv, bh, scope)
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        # Define the proposal Q.
        with tf.variable_scope(self._scopeName, initializer=initializer):
            if Q is not None:
                self._Q = Q
            else:
                logitQ = tf.get_variable(name='logitQ', shape=[dimQ, dimV], initializer=tf.zeros_initializer)
                self._Q = tf.nn.sigmoid(logitQ)
                self._dimQ = dimQ

    """#########################################################################
    sampleVgivenH: the generative direction of the RBM.
    input: H - the latent state, could be [batch, steps, frame]/[batch, frame].
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
        # shape of tensordot = [batch, steps, dimH]/[batch, dimH]
        Pv_h = tf.nn.sigmoid(tf.tensordot(H, tf.transpose(Wt), [[-1], [0]]) + bvt)
        newV = tf.distributions.Bernoulli(probs=Pv_h, dtype=tf.float32).sample()
        return newV, Pv_h

    """#########################################################################
    FreeEnergy: the free energy function.
    input: V - the latent state, could be [batch, steps, dimV]/[batch, dimV].
           W - the weight matrix provided by outer network. Using default one if
               None.
           bv - the visible bias provided by outer network. Using default one if
                None.
           bh - the hidden bias provided by outer network. Using default one if
                None.
    output: the average free energy per bit in tensor.
    #########################################################################"""
    def FreeEnergy(self, V, W=None, bv=None, bh=None):
        Wt = W if W is not None else self._W
        # the shape of bvt could be [dimV] or [batch, steps, dimV] in RNNRBM.
        bvt = bv if bv is not None else self._bv
        # the shape of bvt could be [dimH] or [batch, steps, dimH] in RNNRBM.
        bht = bv if bh is not None else self._bh
        #
        term1 = tf.tensordot(bvt, V, [[-1], [-1]])
        term2 = tf.nn.softplus(tf.tensordot(V, Wt, [[-1], [0]]) + bht)
        return tf.reduce_mean(term1) + tf.reduce_mean(term2)

    """#########################################################################
    LowerBound: define the lower bound of the log-likelihood with given proposal
                Q. If we are going to introduce complicated model like VAE to be
                the proposal. We should define the lower bound outer the class.
    input: V - the latent state, could be [batch, steps, dimV]/[batch, dimV].
           samplesteps - the number of sample to be drawn. Default is 1.
           W - the weight matrix provided by outer network. Using default one if
               None.
           bv - the visible bias provided by outer network. Using default one if
                None.
           bh - the hidden bias provided by outer network. Using default one if
                None.
           Q - the proposal distribution Q provided by outer network. Using 
               default one if None.
    output: the average free energy per bit in tensor.
    #########################################################################"""
    def LowerBound(self, V, samplesteps=1, W=None, bv=None, bh=None, Q=None):
        Wt = W if W is not None else self._W
        # the shape of bvt could be [dimV] or [batch, steps, dimV] in RNNRBM.
        bvt = bv if bv is not None else self._bv
        # the shape of bvt could be [dimH] or [batch, steps, dimH] in RNNRBM.
        bht = bv if bh is not None else self._bh
        # the proposal distribution Q and in the binRBM it's a simple Bernoulli distribution.
        Qt = Q if Q is not None else self._Q
        # generate samples first.
        K = np.random.multinomial(n=1, pvals=[1/self._dimQ]*self._dimQ)
        samples = tf.distributions.Bernoulli(probs=Qt[K]).sample([samplesteps])
        # negative phase/Bit.
        negPhase = self.FreeEnergy(samples, Wt, bvt, bht)
        # positive phase/Bit.
        posPhase = self.FreeEnergy(V, Wt, bvt, bht)



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
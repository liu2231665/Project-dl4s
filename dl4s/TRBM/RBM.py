# TODO: finish ss-rbm.
"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the file contains the model description of RBM. For each
              necessary member functions, the input of system parameters
              is provided. As we are going to build RNNRBM. Those None params
              will be useful at that moment.
              ----2017.11.19
#########################################################################"""

import tensorflow as tf
import numpy as np
from dl4s.tools import BernoulliNLL, GaussNLL

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
           k - Gibbs sampling steps.
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
            k=1
    ):
        # <Tensorflow Session>.
        self._sess = tf.Session()
        # the proposal distribution.
        self._k = k
        # Check whether the configuration is correct.
        if x is not None and x.shape[-1] != dimV:
            raise ValueError("You have provided a input tensor but the last shape is not equal to [dimV]!!")
        if W is not None and (W.shape[0] != dimV or W.shape[1] != dimH):
            raise ValueError("You have provided a W tensor but the shape is not equal to [dimV, dimH]!!")
        if bv is not None and bv.shape[-1] != dimV:
            raise ValueError("You have provided a bv tensor but the last shape is not equal to [dimV]!!")
        if bh is not None and bh.shape[-1] != dimH:
            raise ValueError("You have provided a bh tensor but the last shape is not equal to [dimH]!!")

        self._dimV = dimV
        self._dimH = dimH
        if scope is None:
            self._scopeName = 'RBM'
        else:
            self._scopeName = scope

        # create the system parameters.
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.variable_scope(self._scopeName, initializer=initializer):
            #
            self._W = W if W is not None else tf.get_variable('W', shape=[dimV, dimH])
            self._bv = bv if bv is not None else tf.get_variable('bv', shape=dimV, initializer=tf.zeros_initializer)
            self._bh = bh if bh is not None else tf.get_variable('bh', shape=dimH, initializer=tf.zeros_initializer)
            # if the RBM component is used to build sequential models like RNN-RBM, the input x should be provided as
            # x = [batch, frame]. O.w, we define it as non-temporal data with shape [batch,frame].
            self._V = x if x is not None else tf.placeholder(dtype=tf.float32, shape=[None, dimV], name='V')
            # <tensor placeholder> learning rate.
            self.lr = tf.placeholder(dtype='float32', shape=(), name='learningRate')

    """#########################################################################
    sampleHgivenV: the inference direction of the RBM.
    input: V - the input vector, could be [batch, dimV].
           beta - a scaling factor for AIS.
    output: newH - the new binary latent states sampled from P(H|V).
            Ph_v - the Bernoulli distribution P(H|V), which is also mean(H|V).
    #########################################################################"""
    def sampleHgivenV(self, V, beta=1.0):
        # shape of tensordot = [batch, dimH]
        Ph_v = tf.nn.sigmoid(tf.tensordot(V, beta * self._W, [[-1], [0]]) + self._bh)
        newH = tf.distributions.Bernoulli(probs=Ph_v, dtype=tf.float32).sample()
        return newH, Ph_v

    """#########################################################################
    sampleVgivenH: the generative direction of the RBM.
    input: H - the latent state, could be [batch, dimH].
           beta - a scaling factor for AIS.
    output: new sample and probability depends on the observation form.
    #########################################################################"""
    def sampleVgivenH(self, H, beta=1.0):
        return None, None

    """#########################################################################
    GibbsSampling: Gibbs sampling.
    input: V - the input vector, could be [batch, dimV].
           beta - a scaling factor for AIS.
           k - the running times of the Gibbs sampling. 1 in default.
    output: newV - new sample of V.
            newH - new sample of H.
            Ph_v - P(H|V).
            Pv_h - P(V|H).
    #########################################################################"""
    def GibbsSampling(self, V, beta=1.0, k=1):
        newV = V
        newH = None
        Ph_v = None
        Pv_h = None
        for i in range(k):
            newH, Ph_v = self.sampleHgivenV(newV, beta)
            newV, Pv_h = self.sampleVgivenH(newH, beta)
            if newV is None:
                raise ValueError("You have not yet define the sampleVgivenH!!")
        return newV, newH, Ph_v, Pv_h

    """#########################################################################
    FreeEnergy: the free energy function.
    input: V - the latent state, could be [batch, dimV].
           beta - a scaling factor for AIS.
    output: the average free energy per frame with shape [batch]
    #########################################################################"""
    def FreeEnergy(self, V, beta=1.0):
        return

    """#########################################################################
    ComputeLoss: define the loss of the log-likelihood with given proposal
                Q. If we are going to introduce complicated model like VAE to be
                the proposal. We should define the lower bound outer the class.
    input: V - the latent state, could be [batch, dimV].
           samplesteps - the number of sample to be drawn. Default is 1.
    output: the average loss per frame in tensor.
    #########################################################################"""
    def ComputeLoss(self, V, samplesteps=10):
        # samples.shape = [batch, dimV].
        samples, _, _, _ = self.GibbsSampling(V=V, k=samplesteps)
        samples = tf.stop_gradient(samples)
        # negative phase with shape [samples].
        negPhase = tf.reduce_mean(self.FreeEnergy(samples))
        # positive phase with shape [batch].
        posPhase = tf.reduce_mean(self.FreeEnergy(V))
        return posPhase - negPhase


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
            scope=None,
            k=1,
    ):
        super().__init__(dimV, dimH, init_scale, x, W, bv, bh, scope, k)
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        # Define the proposal Q.
        with tf.variable_scope(self._scopeName, initializer=initializer):
            # pll.
            _, _, _, Pv_h = self.GibbsSampling(self._V, beta=1.0, k=self._k)
            self._pll = dimV * BernoulliNLL(self._V, Pv_h)
            # one step sample.
            self.newV0, self.newH0, self.muH0, self.muV0 = \
                self.GibbsSampling(self._V, k=1)

    """#########################################################################
    sampleVgivenH: the generative direction of the RBM.
    input: H - the latent state, could be [batch, dimH].
           beta - a scaling factor for AIS.
    output: newV - the new sample of the visible units.
            Pv_h - P(V|H).
    #########################################################################"""
    def sampleVgivenH(self, H, beta=1.0):
        # shape of tensordot = [batch, dimH]
        Pv_h = tf.nn.sigmoid(tf.tensordot(H, beta * tf.transpose(self._W), [[-1], [0]]) + self._bv)
        newV = tf.distributions.Bernoulli(probs=Pv_h, dtype=tf.float32).sample()
        return newV, Pv_h

    """#########################################################################
    FreeEnergy: the free energy function.
    input: V - the latent state, could be [batch, dimV].
           beta - a scaling factor for AIS.
    output: the average free energy per frame with shape [batch]
    #########################################################################"""
    def FreeEnergy(self, V, beta=1.0):
        term1 = V * self._bv
        term2 = tf.nn.softplus(tf.tensordot(V, beta * self._W, [[-1], [0]]) + self._bh)
        return -tf.reduce_sum(term1, axis=-1) - tf.reduce_sum(term2, axis=-1)

    """#########################################################################
    AIS: compute the partition function by annealed importance sampling.
    input: run - the number of samples.
           levels - the number of intermediate proposals.
           samplesteps - the number of sample to be drawn. Default is 1.
           Batch - indicate whether the parameter is batch.
           Seq - indicate whether the parameter is batch.
    output: the log partition function logZB in tensor.
    #########################################################################"""
    def AIS(self, run=10, levels=10, Batch=None, Seq=None):
        # proposal partition function with shape []/[...].
        logZA = tf.reduce_sum(tf.nn.softplus(self._bv), axis=-1) + \
                tf.reduce_sum(tf.nn.softplus(self._bh), axis=-1)
        if Batch is not None and Seq is None:
            sample = tf.zeros(shape=[run, Batch, self._dimV])
        elif Batch is None and Seq is not None:
            sample = tf.zeros(shape=[run, Seq, self._dimV])
        elif Batch is not None and Seq is not None:
            sample = tf.zeros(shape=[run, Batch, Seq, self._dimV])
        else:
            sample = tf.zeros(shape=[run, self._dimV])
        # Define the intermediate levels.
        betas = np.random.rand(levels)
        betas.sort()
        sample, _, _, _ = self.GibbsSampling(V=sample, beta=0.0)
        # logwk is the weighted matrix.
        logwk = tf.zeros(shape=tf.shape(sample)[0:-1], dtype=tf.float32)
        for i in range(len(betas)):
            sample, _, _, _ = self.GibbsSampling(V=sample, beta=betas[i])
            # logp_k, logp_km1 shape [run, ...]
            logp_k = -self.FreeEnergy(V=sample, beta=betas[i])
            if i != 0:
                logp_km1 = -self.FreeEnergy(V=sample, beta=betas[i-1])
            else:
                logp_km1 = -self.FreeEnergy(V=sample, beta=0.0)
            logwk += logp_k - logp_km1
        # beta = 1.0
        sample, _, _, _ = self.GibbsSampling(V=sample, beta=1.0)
        logp_k = -self.FreeEnergy(V=sample, beta=1.0)
        logp_km1 = -self.FreeEnergy(V=sample, beta=betas[-1])
        logwk += logp_k - logp_km1

        # compute the average weight. [...]
        log_wk_mean = tf.reduce_mean(logwk, axis=0)
        r_ais = tf.reduce_mean(tf.exp(logwk-log_wk_mean), axis=0)
        return logZA + tf.log(r_ais) + log_wk_mean


"""#########################################################################
Class: gaussRBM - the RBM model for continous observations
#########################################################################"""
class gaussRBM(_RBM, object):
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
           W_Norm - whether constraint the W by norm.
           bv - the visible bias that maybe provided by outer parts. None in
               default and the RBM will define its own visible bias.
           bh - the hidden bias that maybe provided by outer parts. None in
               default and the RBM will define its own hidden bias.
           std - the standard deviation parameter of the Gaussian RBM. None in
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
            std=None,
            scope=None,
            k=1,
    ):
        super().__init__(dimV, dimH, init_scale, x, W, bv, bh, scope, k)
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        # Define the proposal Q.
        with tf.variable_scope(self._scopeName, initializer=initializer):
            # define the std parameter.
            self._std = std if std is not None else tf.get_variable('std', shape=dimV, initializer=tf.zeros_initializer)
            # pll.
            _, _, _, muV = self.GibbsSampling(self._V, beta=1.0, k=self._k)
            self._monitor = tf.reduce_mean(tf.reduce_sum((self._V - muV)**2, axis=[-1]))
            # one step sample.
            self.newV0, self.newH0, self.muH0, self.muV0 = \
                self.GibbsSampling(self._V, k=1)

    """#########################################################################
    sampleHgivenV: the inference direction of the RBM.
    input: V - the input vector, could be [batch, dimV].
           beta - a scaling factor for AIS.
    output: newH - the new binary latent states sampled from P(H|V).
            Ph_v - the Bernoulli distribution P(H|V), which is also mean(H|V).
    #########################################################################"""
    def sampleHgivenV(self, V, beta=1.0):
        # shape of tensordot = [batch, dimH]
        Ph_v = tf.nn.sigmoid(tf.tensordot(V/(tf.nn.softplus(self._std)**2 + 1e-8), beta * self._W, [[-1], [0]]) + self._bh)
        newH = tf.distributions.Bernoulli(probs=Ph_v, dtype=tf.float32).sample()
        return newH, Ph_v

    """#########################################################################
    sampleVgivenH: the generative direction of the RBM.
    input: H - the latent state, could be [batch, dimH].
           beta - a scaling factor for AIS.
    output: muV - the new sample of the visible units.
            newV - noisy sample of the visible units.
    #########################################################################"""
    def sampleVgivenH(self, H, beta=1.0):
        # shape of tensordot = [batch, dimH]
        muV = tf.tensordot(H, beta * tf.transpose(self._W), [[-1], [0]]) + self._bv
        return muV, muV

    """#########################################################################
    FreeEnergy: the free energy function.
    input: V - the latent state, could be [batch, dimV].
           beta - a scaling factor for AIS.
    output: the average free energy per frame with shape [batch]
    #########################################################################"""
    def FreeEnergy(self, V, beta=1.0):
        #
        term1 = (V - self._bv)**2 / (2 * tf.nn.softplus(self._std)**2 + 1e-8)
        term2 = tf.nn.softplus(tf.tensordot(V/(tf.nn.softplus(self._std)**2 + 1e-8), beta * self._W, [[-1], [0]]) + self._bh)
        return tf.reduce_sum(term1, axis=-1) - tf.reduce_sum(term2, axis=-1)

    """#########################################################################
    AIS: compute the partition function by annealed importance sampling.
    input: run - the number of samples.
           levels - the number of intermediate proposals.
           samplesteps - the number of sample to be drawn. Default is 1.
           Batch - indicate whether the parameter is batch.
           Seq - indicate whether the parameter is batch.
    output: the log partition function logZB in tensor.
    #########################################################################"""
    def AIS(self, run=10, levels=10, Batch=None, Seq=None):
        # proposal partition function with shape []/[...].
        logZA_term1 = 0.5 * tf.log(2*np.pi) + tf.log(tf.nn.softplus(self._std))
        logZA = tf.reduce_sum(logZA_term1, axis=-1) + \
                tf.reduce_sum(tf.nn.softplus(self._bh), axis=-1)
        if Batch is not None and Seq is None:
            sample = tf.zeros(shape=[run, Batch, self._dimV])
        elif Batch is None and Seq is not None:
            sample = tf.zeros(shape=[run, Seq, self._dimV])
        elif Batch is not None and Seq is not None:
            sample = tf.zeros(shape=[run, Batch, Seq, self._dimV])
        else:
            sample = tf.zeros(shape=[run, self._dimV])
        # Define the intermediate levels.
        betas = np.random.rand(levels)
        betas.sort()
        sample, _, _, _ = self.GibbsSampling(V=sample, beta=0.0)
        # logwk is the weighted matrix.
        logwk = tf.zeros(shape=tf.shape(sample)[0:-1], dtype=tf.float32)
        for i in range(len(betas)):
            sample, _, _, _ = self.GibbsSampling(V=sample, beta=betas[i])
            # logp_k, logp_km1 shape [run, ...]
            logp_k = -self.FreeEnergy(V=sample, beta=betas[i])
            if i != 0:
                logp_km1 = -self.FreeEnergy(V=sample, beta=betas[i - 1])
            else:
                logp_km1 = -self.FreeEnergy(V=sample, beta=0.0)
            logwk += logp_k - logp_km1
        # beta = 1.0
        sample, _, _, _ = self.GibbsSampling(V=sample, beta=1.0)
        logp_k = -self.FreeEnergy(V=sample, beta=1.0)
        logp_km1 = -self.FreeEnergy(V=sample, beta=betas[-1])
        logwk += logp_k - logp_km1

        # compute the average weight. [...]
        log_wk_mean = tf.reduce_mean(logwk, axis=0)
        r_ais = tf.reduce_mean(tf.exp(logwk - log_wk_mean), axis=0)
        return logZA + tf.log(r_ais) + log_wk_mean



"""#########################################################################
Class: mu_ssRBM - the mu-pooled spike and slab Restricted Boltzmann Machine 
                  for continuous observations.
#########################################################################"""
class mu_ssRBM(object):
    """#########################################################################
    __init__:the initialization function.
    input: dimV - the frame dimension of the input vector.
           dimH - the frame dimension of the latent state.
           dimS - the frame dimension of S. (useless in current version)
           init_scale - the initial scale of system parameters.
           x - the input tensor that maybe provided by outer parts. None in
               default and the RBM will define its own input tensor.
           W - the weight matrix that maybe provided by outer parts. None in
               default and the RBM will define its own input tensor.
           bv - the visible bias that maybe provided by outer parts. None in
               default and the RBM will define its own visible bias.
           bh - the hidden bias that maybe provided by outer parts. None in
               default and the RBM will define its own hidden bias.
           gamma - the precision parameter for V (Lambda in ICML paper.)
           alpha - the precision parameter for S.
           alphaTrain - indicate whether train alpha or set as hyper parameter.
           mu - the mean parameter for S.
           muTrain - indicate whether train mu or set as hyper parameter.
           phi - the precision parameter of V modulated by H.
           phiTrain - indicate whether train phi or set as hyper parameter.
           scope - using to define the variable_scope.
           k - Gibbs sampling steps.
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
            dimS=None,
            gamma=None,
            alpha=None,
            alphaTrain=False,
            mu=None,
            muTrain=False,
            phi=None,
            phiTrain=False,
            scope=None,
            bound=[-25.0, 25.0],
            k=1
    ):
        self._sess = tf.Session()
        # the proposal distribution.
        self._k = k
        #
        self._bound = bound
        # Check whether the configuration is correct.
        if x is not None and x.shape[-1] != dimV:
            raise ValueError("You have provided a input tensor but the last shape is not equal to [dimV]!!")
        if W is not None and (W.shape[0] != dimV or W.shape[1] != dimH):
            raise ValueError("You have provided a W tensor but the shape is not equal to [dimV, dimH]!!")
        if bv is not None and bv.shape[-1] != dimV:
            raise ValueError("You have provided a bv tensor but the last shape is not equal to [dimV]!!")
        if bh is not None and bh.shape[-1] != dimH:
            raise ValueError("You have provided a bh tensor but the last shape is not equal to [dimH]!!")
        if phi is not None and (phi.shape[0] != dimH or phi.shape[1] != dimV):
            raise ValueError("You have provided a phi tensor but the shape is not equal to [dimH, dimV]!!")

        self._dimV = dimV
        self._dimH = dimH
        self._dimS = dimS if dimS is not None else dimH
        if scope is None:
            self._scopeName = 'ssRBM'
        else:
            self._scopeName = scope
        #
        if gamma is not None and gamma.shape[-1] != self._dimV:
            raise ValueError("You have provided a gamma tensor but the last shape is not equal to [dimV]!!")
        if alpha is not None and alpha.shape[-1] != self._dimS:
            raise ValueError("You have provided a alpha tensor but the last shape is not equal to [dimS]!!")
        if mu is not None and mu.shape[-1] != self._dimS:
            raise ValueError("You have provided a alpha tensor but the last shape is not equal to [dimS]!!")

        # create the system parameters.
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.variable_scope(self._scopeName, initializer=initializer):
            #
            self._W = W if W is not None else tf.get_variable('W', shape=[dimV, dimH])
            self._bv = bv if bv is not None else tf.get_variable('bv', shape=dimV, initializer=tf.zeros_initializer)
            self._bh = bh if bh is not None else tf.get_variable('bh', shape=dimH, initializer=tf.zeros_initializer)
            #
            self._gamma = gamma if gamma is not None else \
                tf.nn.softplus(tf.get_variable('gamma', shape=self._dimV, initializer=tf.zeros_initializer))
            if alpha is not None:
                self._alpha = alpha
            else:
                if alphaTrain:
                    self._alpha = tf.exp(tf.get_variable('alpha', shape=self._dimS,
                                                             initializer=tf.zeros_initializer))
                else:
                    self._alpha = tf.ones(shape=self._dimS, name='alpha')
            #
            if mu is not None:
                self._mu = mu
            else:
                if muTrain:
                    self._mu = tf.get_variable('mu', shape=self._dimS, initializer=tf.zeros_initializer)
                else:
                    self._mu = tf.zeros(shape=self._dimS, name='mu')
            #
            if phi is not None:
                self._phi = phi
            else:
                if phiTrain:
                    self._phi = tf.nn.relu(tf.get_variable('phi', shape=(self._dimH, self._dimV)))
                else:
                    self._phi = tf.zeros(shape=(self._dimH, self._dimV), name='phi')
            # x = [batch, steps, dimV]. O.w, we define it as non-temporal data with shape [batch,dimV].
            self._V = x if x is not None else tf.placeholder(dtype=tf.float32, shape=[None, dimV], name='V')
            # <tensor placeholder> learning rate.
            self.lr = tf.placeholder(dtype='float32', shape=(), name='learningRate')
            self.newV, self.newH, self.newS, self.muV, self.muH, self.muS = \
                self.GibbsSampling(self._V, k=k)
            newH = tf.expand_dims(self.newH, axis=2)
            W = tf.expand_dims(tf.expand_dims(self._W, axis=0), axis=0)
            term1 = newH * W / (self._alpha + 1e-8)
            term1 = tf.tensordot(term1, self._W, [[-1], [-1]])
            # define the covariance and precision.
            Cv_sh = 1 / (self._gamma + tf.tensordot(newH, self._phi, [[-1], [0]]) + 1e-8)
            term2 = Cv_sh * tf.eye(self._dimV, batch_shape=[1, 1])
            self.PreV_h = term2 + term1
            self.CovV_h = tf.matrix_inverse(self.PreV_h)
            # define the monitor.
            monitor = tf.reduce_sum((self._V - self.muV) ** 2, axis=-1)
            # one step sample.
            self.newV0, self.newH0, self.newS0, self.muV0, self.muH0, self.muS0 = \
                self.GibbsSampling(self._V, k=1)
            self._monitor = tf.sqrt(tf.reduce_mean(monitor))
            #self._monitor = muV


    """#########################################################################
    sampleHgivenV: the inference direction of the RBM by P(H|V).
    input: V - the input vector, could be [batch, dimV].
           beta - a scaling factor for AIS.
    output: newH - the new binary latent states sampled from P(H|V).
            meanH_v - the Bernoulli distribution P(H|V), which is also mean(H|V).
    #########################################################################"""
    def sampleHgivenV(self, V, beta=1.0):
        factorV = tf.tensordot(V, beta * self._W, [[-1], [0]])  # shape = [..., dimH]
        sqr_term = (0.5 * factorV ** 2) / (self._alpha + 1e-8) \
                   - 0.5 * tf.tensordot(V**2, tf.transpose(self._phi), [[-1], [0]])
        lin_term = factorV * self._mu
        if beta == 0.0:
            meanH_v = tf.nn.sigmoid(self._bh, name='meanH_v')
        else:
            meanH_v = tf.nn.sigmoid(sqr_term + lin_term + self._bh, name='meanH_v')
        newH = tf.distributions.Bernoulli(probs=meanH_v, dtype=tf.float32).sample()
        return newH, meanH_v

    """#########################################################################
    sampleSgivenVH: the other inference direction of the RBM by P(S|V, H).
    input: V - the input vector, could be [batch, dimV].
           H - the spike vector, could be [batch, dimH].
           beta - a scaling factor for AIS.
    output: newS - the new slab latent states sampled from P(S|V, H).
            meanS_vh - the meanS given V, H.
    #########################################################################"""
    def sampleSgivenVH(self, V, H, beta=1.0):
        #
        factorV = tf.tensordot(V, beta * self._W, [[-1], [0]]) / (self._alpha + 1e-8)\
                  + self._mu  # shape = [..., dimH]
        meanS_vh1 = factorV
        eps = tf.truncated_normal(shape=(tf.shape(meanS_vh1)))
        if beta == 0.0:
            newS = meanS_vh1 * H
        else:
            newS = meanS_vh1 * H + tf.sqrt(self._alpha) * eps
        return newS, meanS_vh1

    """#########################################################################
    sampleVgivenSH: the generative direction of the RBM by P(V|S, H).
    input: S - the slab vector, could be [..., dimH].
           H - the spike vector, could be [..., dimH].
           beta - a scaling factor for AIS.
    output: newV_sh - the new slab latent states sampled from P(V|S, H).
            meanV_sh - the meanV given S, H.
    #########################################################################"""
    def sampleVgivenSH(self, S, H, beta=1.0):
        # shape = [..., dimV]
        if beta == 0.0:
            Cv_sh = 1 / (self._gamma + 1e-8)
        else:
            Cv_sh = 1 / (self._gamma + tf.tensordot(H, self._phi, [[-1], [0]]) + 1e-8)
        # shape = [..., dimV]
        meanV_sh = Cv_sh * (tf.tensordot(S*H, beta * tf.transpose(self._W), [[-1], [0]])
                            + self._bv)
        eps = tf.truncated_normal(shape=(tf.shape(meanV_sh)))
        newV_sh = meanV_sh + tf.sqrt(Cv_sh) * eps
        #return newV_sh, meanV_sh
        return newV_sh, meanV_sh

    """#########################################################################
    GibbsSampling: Gibbs sampling.
    input: V - the input vector, could be [batch, dimV].
           beta - a scaling factor for AIS.
           k - the running times of the Gibbs sampling. 1 in default.
    output: 
    #########################################################################"""
    def GibbsSampling(self, V, beta=1.0, k=1):
        newV = V
        newH = None
        newS = None
        meanH_v = None
        meanS_vh = None
        meanV_sh = None
        if k < 1:
            raise ValueError("k should be greater than zero!!")
        for i in range(k):
            newH, meanH_v = self.sampleHgivenV(newV, beta)
            newS, meanS_vh = self.sampleSgivenVH(newV, newH, beta)
            newV, meanV_sh = self.sampleVgivenSH(newS, newH, beta)
        return newV, newH, newS, meanV_sh, meanH_v, meanS_vh

    """#########################################################################
    FreeEnergy: the free energy function.
    input: V - the latent state, could be [batch, dimV].
           beta - a scaling factor for AIS.
    output: the average free energy per frame with shape [batch]
    #########################################################################"""
    def FreeEnergy(self, V, beta=1.0):
        sqr_term = 0.5 * tf.tensordot(V**2, self._gamma, [[-1], [0]])
        lin_term = tf.tensordot(V, self._bv, [[-1], [0]])
        con_term = 0.5 * tf.reduce_sum(tf.log(2*np.pi) - tf.log(self._alpha + 1e-8))
        #
        factorV = tf.tensordot(V, beta * self._W, [[-1], [0]])  # shape = [..., dimH]
        sqr_term_h = (0.5 * factorV ** 2) / (self._alpha + 1e-8) \
                   - 0.5 * tf.tensordot(V ** 2, tf.transpose(self._phi), [[-1], [0]])
        lin_term_h = factorV * self._mu
        if beta == 0.0:
            splus_term = tf.reduce_sum(tf.nn.softplus(self._bh),
                                      axis=[-1])
        else:
            splus_term = tf.reduce_sum(tf.nn.softplus(sqr_term_h + lin_term_h + self._bh),
                                       axis=[-1])
        return sqr_term - splus_term - lin_term - con_term

    """#########################################################################
    ComputeLoss: define the loss of the log-likelihood with given proposal
                Q. If we are going to introduce complicated model like VAE to be
                the proposal. We should define the lower bound outer the class.
    input: V - the latent state, could be [..., dimV].
           samplesteps - the number of sample to be drawn. Default is 1.
    output: the average loss per frame in tensor.
    #########################################################################"""
    def ComputeLoss(self, V, samplesteps=10):
        # samples.shape = [..., dimV].
        samples, _, _, _, _, _ = self.GibbsSampling(V=V, k=samplesteps)
        samples = tf.stop_gradient(samples)
        # negative phase with shape [samples].
        negPhase = tf.reduce_mean(self.FreeEnergy(samples))
        # positive phase with shape [batch].
        posPhase = tf.reduce_mean(self.FreeEnergy(V))
        return posPhase - negPhase

    """#########################################################################
    AIS: compute the partition function by annealed importance sampling.
    input: run - the number of samples.
           levels - the number of intermediate proposals.
           samplesteps - the number of sample to be drawn. Default is 1.
           Batch - indicate whether the parameter is batch.
           Seq - indicate whether the parameter is batch.
    output: the log partition function logZB in tensor.
    #########################################################################"""
    def AIS(self, run=10, levels=10, Batch=None, Seq=None):
        # proposal partition function with shape []/[...].
        logZA_term1 = 0.5 * tf.tensordot(self._bv**2, 1/(self._gamma+1e-8), [[-1], [0]])
        logZA_term2 = 0.5 * tf.reduce_sum(tf.log(2*np.pi/(self._gamma+1e-8)), axis=[-1])
        logZA_term3 = 0.5 * self._dimV * tf.reduce_sum(tf.log(2*np.pi) - tf.log(1e-8))
        logZA_term4 = tf.reduce_sum(tf.nn.softplus(self._bh), axis=-1)
        logZA = logZA_term1 + logZA_term2 + logZA_term3 + logZA_term4

        if Batch is not None and Seq is None:
            sample = tf.zeros(shape=[run, Batch, self._dimV])
        elif Batch is None and Seq is not None:
            sample = tf.zeros(shape=[run, Seq, self._dimV])
        elif Batch is not None and Seq is not None:
            sample = tf.zeros(shape=[run, Batch, Seq, self._dimV])
        else:
            sample = tf.zeros(shape=[run, self._dimV])
        # Define the intermediate levels.
        betas = 0.5 * np.random.rand(levels) + 0.5
        betas.sort()
        sample, _, _, _, _, _ = self.GibbsSampling(V=sample, beta=0.0)
        sample = tf.maximum(self._bound[0], sample)
        sample = tf.minimum(self._bound[1], sample)
        # logwk is the weighted matrix.
        logwk = tf.zeros(shape=tf.shape(sample)[0:-1], dtype=tf.float32)
        for i in range(len(betas)):
            sample, _, _, _, _, _ = self.GibbsSampling(V=sample, beta=betas[i])
            sample = tf.maximum(self._bound[0], sample)
            sample = tf.minimum(self._bound[1], sample)
            # logp_k, logp_km1 shape [run, ...]
            logp_k = -self.FreeEnergy(V=sample, beta=betas[i])
            if i != 0:
                logp_km1 = -self.FreeEnergy(V=sample, beta=betas[i - 1])
            else:
                logp_km1 = -self.FreeEnergy(V=sample, beta=0.0)
            logwk += logp_k - logp_km1
        # beta = 1.0
        sample, _, _, _, _, _ = self.GibbsSampling(V=sample, beta=1.0)
        sample = tf.maximum(self._bound[0], sample)
        sample = tf.minimum(self._bound[1], sample)
        logp_k = -self.FreeEnergy(V=sample, beta=1.0)
        logp_km1 = -self.FreeEnergy(V=sample, beta=betas[-1])
        logwk += logp_k - logp_km1

        # compute the average weight. [...]
        log_wk_mean = tf.reduce_max(logwk, axis=0)
        r_ais = tf.reduce_mean(tf.exp(logwk - log_wk_mean), axis=0)
        #return sample
        return logZA + log_wk_mean + tf.log(r_ais)

    """#########################################################################
    add_constraint: compute the partition function by annealed importance sampling.
    input: .
    output: the tensor that assign the normalization to W.
    #########################################################################"""
    def add_constraint(self):
        Wnorm = tf.stop_gradient(tf.norm(self._W, axis=0))
        #mask = tf.minimum(1.0, Wnorm)
        return tf.assign(self._W, self._W / Wnorm)


"""#########################################################################
Class: bin_ssRBM - the mu-pooled spike and slab Restricted Boltzmann Machine 
                  for binary observations.
#########################################################################"""
class bin_ssRBM(object):
    """#########################################################################
    __init__:the initialization function.
    input: dimV - the frame dimension of the input vector.
           dimH - the frame dimension of the latent state.
           dimS - the frame dimension of S. (useless in current version)
           init_scale - the initial scale of system parameters.
           x - the input tensor that maybe provided by outer parts. None in
               default and the RBM will define its own input tensor.
           W - the weight matrix that maybe provided by outer parts. None in
               default and the RBM will define its own input tensor.
           bv - the visible bias that maybe provided by outer parts. None in
               default and the RBM will define its own visible bias.
           bh - the hidden bias that maybe provided by outer parts. None in
               default and the RBM will define its own hidden bias.
           alpha - the precision parameter for S.
           alphaTrain - indicate whether train alpha or set as hyper parameter.
           mu - the mean parameter for S.
           muTrain - indicate whether train mu or set as hyper parameter.
           scope - using to define the variable_scope.
           k - Gibbs sampling steps.
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
            dimS=None,
            alpha=None,
            alphaTrain=False,
            mu=None,
            muTrain=False,
            scope=None,
            k=1
    ):
        self._sess = tf.Session()
        # the proposal distribution.
        self._k = k
        # Check whether the configuration is correct.
        if x is not None and x.shape[-1] != dimV:
            raise ValueError("You have provided a input tensor but the last shape is not equal to [dimV]!!")
        if W is not None and (W.shape[0] != dimV or W.shape[1] != dimH):
            raise ValueError("You have provided a W tensor but the shape is not equal to [dimV, dimH]!!")
        if bv is not None and bv.shape[-1] != dimV:
            raise ValueError("You have provided a bv tensor but the last shape is not equal to [dimV]!!")
        if bh is not None and bh.shape[-1] != dimH:
            raise ValueError("You have provided a bh tensor but the last shape is not equal to [dimH]!!")

        self._dimV = dimV
        self._dimH = dimH
        self._dimS = dimS if dimS is not None else dimH
        if scope is None:
            self._scopeName = 'ssRBM'
        else:
            self._scopeName = scope
        #
        if alpha is not None and alpha.shape[-1] != self._dimS:
            raise ValueError("You have provided a alpha tensor but the last shape is not equal to [dimS]!!")
        if mu is not None and mu.shape[-1] != self._dimS:
            raise ValueError("You have provided a alpha tensor but the last shape is not equal to [dimS]!!")
        # create the system parameters.
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.variable_scope(self._scopeName, initializer=initializer):
            #
            self._W = W if W is not None else tf.get_variable('W', shape=[dimV, dimH])
            self._bv = bv if bv is not None else tf.get_variable('bv', shape=dimV, initializer=tf.zeros_initializer)
            self._bh = bh if bh is not None else tf.get_variable('bh', shape=dimH, initializer=tf.zeros_initializer)
            #
            if alpha is not None:
                self._alpha = alpha
            else:
                if alphaTrain:
                    self._alpha = tf.exp(tf.get_variable('alpha', shape=self._dimS,
                                                         initializer=tf.zeros_initializer))
                else:
                    self._alpha = tf.ones(shape=self._dimS, name='alpha')
            #
            if mu is not None:
                self._mu = mu
            else:
                if muTrain:
                    self._mu = tf.get_variable('mu', shape=self._dimS, initializer=tf.zeros_initializer)
                else:
                    self._mu = tf.zeros(shape=self._dimS, name='mu')
            #
            # x = [batch, steps, dimV]. O.w, we define it as non-temporal data with shape [batch,dimV].
            self._V = x if x is not None else tf.placeholder(dtype=tf.float32, shape=[None, dimV], name='V')
            # <tensor placeholder> learning rate.
            self.lr = tf.placeholder(dtype='float32', shape=(), name='learningRate')
            self.newV, self.newH, self.newS, self.muV, self.muH, self.muS = \
                self.GibbsSampling(self._V, k=k)
            # define the monitor.
            self._monitor = BernoulliNLL(self._V, self.muV) * dimV
            # one step sample.
            self.newV0, self.newH0, self.newS0, self.muV0, self.muH0, self.muS0 = \
                self.GibbsSampling(self._V, k=1)

    """#########################################################################
    sampleSgivenVH: the other inference direction of the RBM by P(S|V, H).
    input: V - the input vector, could be [batch, dimV].
           H - the spike vector, could be [batch, dimH].
           beta - a scaling factor for AIS.
    output: newS - the new slab latent states sampled from P(S|V, H).
            meanS_vh - the meanS given V, H.
    #########################################################################"""
    def sampleSgivenVH(self, V, H, beta=1.0):
        #
        factorV = tf.tensordot(V, beta * self._W, [[-1], [0]]) / (self._alpha + 1e-8) + self._mu  # shape = [..., dimH]
        meanS_vh1 = factorV
        eps = tf.truncated_normal(shape=(tf.shape(meanS_vh1)))
        newS = meanS_vh1 * H + tf.sqrt(self._alpha) * eps
        return newS, meanS_vh1

    """#########################################################################
    sampleHgivenV: the inference direction of the RBM by P(H|V).
    input: V - the input vector, could be [batch, dimV].
           beta - a scaling factor for AIS.
    output: newH - the new binary latent states sampled from P(H|V).
            meanH_v - the Bernoulli distribution P(H|V), which is also mean(H|V).
    #########################################################################"""
    def sampleHgivenV(self, V, beta=1.0):
        factorV = tf.tensordot(V, beta * self._W, [[-1], [0]])  # shape = [..., dimH]
        sqr_term = (0.5 * factorV ** 2) / (self._alpha + 1e-8)
        lin_term = factorV * self._mu
        meanH_v = tf.nn.sigmoid(sqr_term + lin_term + self._bh, name='meanH_v')
        newH = tf.distributions.Bernoulli(probs=meanH_v, dtype=tf.float32).sample()
        return newH, meanH_v

    """#########################################################################
    sampleVgivenSH: the generative direction of the RBM by P(V|S, H).
    input: S - the slab vector, could be [..., dimH].
           H - the spike vector, could be [..., dimH].
           beta - a scaling factor for AIS.
    output: newV_sh - the new slab latent states sampled from P(V|S, H).
            meanV_sh - the meanV given S, H.
    #########################################################################"""
    def sampleVgivenSH(self, S, H, beta=1.0):
        probit =tf.tensordot(S * H, beta * tf.transpose(self._W), [[-1], [0]]) + self._bv
        meanV_sh = tf.nn.sigmoid(probit, name='meanH_v')
        newV_sh = tf.distributions.Bernoulli(probs=meanV_sh, dtype=tf.float32).sample()
        # return newV_sh, meanV_sh
        return newV_sh, meanV_sh

    """#########################################################################
    GibbsSampling: Gibbs sampling.
    input: V - the input vector, could be [batch, dimV].
           beta - a scaling factor for AIS.
           k - the running times of the Gibbs sampling. 1 in default.
    output: 
    #########################################################################"""
    def GibbsSampling(self, V, beta=1.0, k=1):
        newV = V
        newH = None
        newS = None
        meanH_v = None
        meanS_vh = None
        meanV_sh = None
        if k < 1:
            raise ValueError("k should be greater than zero!!")
        for i in range(k):
            newH, meanH_v = self.sampleHgivenV(newV, beta)
            newS, meanS_vh = self.sampleSgivenVH(newV, newH, beta)
            newV, meanV_sh = self.sampleVgivenSH(newS, newH, beta)
        return newV, newH, newS, meanV_sh, meanH_v, meanS_vh

    """#########################################################################
    FreeEnergy: the free energy function.
    input: V - the latent state, could be [batch, dimV].
           beta - a scaling factor for AIS.
    output: the average free energy per frame with shape [batch]
    #########################################################################"""
    def FreeEnergy(self, V, beta=1.0):
        #
        lin_term = tf.tensordot(V, self._bv, [[-1], [0]])
        con_term = 0.5 * tf.reduce_sum(tf.log(2 * np.pi) - tf.log(self._alpha + 1e-8))
        #
        factorV = tf.tensordot(V, beta * self._W, [[-1], [0]])  # shape = [..., dimH]
        sqr_term_h = (0.5 * factorV ** 2) / (self._alpha + 1e-8)
        lin_term_h = factorV * self._mu
        splus_term = tf.reduce_sum(tf.nn.softplus(sqr_term_h + lin_term_h + self._bh),
                                   axis=[-1])
        return -splus_term - lin_term - con_term

    """#########################################################################
    ComputeLoss: define the loss of the log-likelihood with given proposal
                Q. If we are going to introduce complicated model like VAE to be
                the proposal. We should define the lower bound outer the class.
    input: V - the latent state, could be [..., dimV].
           samplesteps - the number of sample to be drawn. Default is 1.
    output: the average loss per frame in tensor.
    #########################################################################"""
    def ComputeLoss(self, V, samplesteps=10):
        # samples.shape = [..., dimV].
        samples, _, _, _, _, _ = self.GibbsSampling(V=V, k=samplesteps)
        samples = tf.stop_gradient(samples)
        # negative phase with shape [samples].
        negPhase = tf.reduce_mean(self.FreeEnergy(samples))
        # positive phase with shape [batch].
        posPhase = tf.reduce_mean(self.FreeEnergy(V))
        return posPhase - negPhase

    """#########################################################################
    AIS: compute the partition function by annealed importance sampling.
    input: run - the number of samples.
           levels - the number of intermediate proposals.
           samplesteps - the number of sample to be drawn. Default is 1.
           Batch - indicate whether the parameter is batch.
           Seq - indicate whether the parameter is batch.
    output: the log partition function logZB in tensor.
    #########################################################################"""
    def AIS(self, run=10, levels=10, Batch=None, Seq=None):
        # proposal partition function with shape []/[...].
        logZA_term1 = tf.reduce_sum(tf.nn.softplus(self._bv), axis=-1)
        logZA_term2 = tf.reduce_sum(tf.nn.softplus(self._bh), axis=-1)
        logZA = logZA_term1 + logZA_term2

        if Batch is not None and Seq is None:
            sample = tf.zeros(shape=[run, Batch, self._dimV])
        elif Batch is None and Seq is not None:
            sample = tf.zeros(shape=[run, Seq, self._dimV])
        elif Batch is not None and Seq is not None:
            sample = tf.zeros(shape=[run, Batch, Seq, self._dimV])
        else:
            sample = tf.zeros(shape=[run, self._dimV])
        # Define the intermediate levels.
        betas = np.random.rand(levels)
        betas.sort()
        sample, _, _, _, _, _ = self.GibbsSampling(V=sample, beta=0.0)
        # logwk is the weighted matrix.
        logwk = tf.zeros(shape=tf.shape(sample)[0:-1], dtype=tf.float32)
        for i in range(len(betas)):
            sample, _, _, _, _, _ = self.GibbsSampling(V=sample, beta=betas[i])
            # logp_k, logp_km1 shape [run, ...]
            logp_k = -self.FreeEnergy(V=sample, beta=betas[i])
            if i != 0:
                logp_km1 = -self.FreeEnergy(V=sample, beta=betas[i - 1])
            else:
                logp_km1 = -self.FreeEnergy(V=sample, beta=0.0)
            logwk += logp_k - logp_km1
        # beta = 1.0
        sample, _, _, _, _, _ = self.GibbsSampling(V=sample, beta=1.0)
        logp_k = -self.FreeEnergy(V=sample, beta=1.0)
        logp_km1 = -self.FreeEnergy(V=sample, beta=betas[-1])
        logwk += logp_k - logp_km1

        # compute the average weight. [...]
        log_wk_mean = tf.reduce_mean(logwk, axis=0)
        r_ais = tf.reduce_mean(tf.exp(logwk - log_wk_mean), axis=0)
        return logZA + tf.log(r_ais) + log_wk_mean

    """#########################################################################
    add_constraint: compute the partition function by annealed importance sampling.
    input: .
    output: the tensor that assign the normalization to W.
    #########################################################################"""
    def add_constraint(self):
        Wnorm = tf.stop_gradient(tf.norm(self._W, axis=0))
        # mask = tf.minimum(1.0, Wnorm)
        return tf.assign(self._W, self._W / Wnorm)
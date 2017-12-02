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
        if W is not None and W.shape[0] != dimV and W.shape[1] !=dimH:
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
            sample = tf.ones(shape=[run, Batch, self._dimV])
        elif Batch is None and Seq is not None:
            sample = tf.ones(shape=[run, Seq, self._dimV])
        elif Batch is not None and Seq is not None:
            sample = tf.ones(shape=[run, Batch, Seq, self._dimV])
        else:
            sample = tf.ones(shape=[run, self._dimV])
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
    output: newV - the new sample of the visible units.
            Pv_h - P(V|H).
    #########################################################################"""
    def sampleVgivenH(self, H, beta=1.0):
        # shape of tensordot = [batch, dimH]
        muV = tf.tensordot(H, beta * tf.transpose(self._W), [[-1], [0]]) + self._bv
        newV = tf.distributions.Normal(loc=muV, scale=tf.nn.softplus(self._std)).sample()
        return newV, muV

    """#########################################################################
    FreeEnergy: the free energy function.
    input: V - the latent state, could be [batch, dimV].
           beta - a scaling factor for AIS.
    output: the average free energy per frame with shape [batch]
    #########################################################################"""
    def FreeEnergy(self, V, beta=1.0):
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
            sample = tf.ones(shape=[run, Batch, self._dimV])
        elif Batch is None and Seq is not None:
            sample = tf.ones(shape=[run, Seq, self._dimV])
        elif Batch is not None and Seq is not None:
            sample = tf.ones(shape=[run, Batch, Seq, self._dimV])
        else:
            sample = tf.ones(shape=[run, self._dimV])
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
Class: _ssRBM - the spike and slab Restricted Boltzmann Machine.
#########################################################################"""
class _ssRBM(object):
    pass
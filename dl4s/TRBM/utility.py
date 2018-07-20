"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the file contains the utility of RBM including Gibbs sampling,
              annealed importance sampling.
              ----2017.11.03
#########################################################################"""
from dl4s.cores.model import _config

"""#########################################################################
Class: configRNNRBM - Basic setting of the RNN-RBM models. 
#########################################################################"""
class configRNNRBM(_config):
    aisRun = 100        # <scalar> the number of samples of AIS.
    aisLevel = 10       # <scalar> the number of intermediate proposal distributions of AIS.
    Gibbs = 15          # <scalar> the steps of Gibbs sampling.
    recType = 'LSTM'
    mlpType = 'relu'
    dimMlp = []
    dimRec = []
    dimState = 100      # <scalar> the size of the stochastic layer.


"""#########################################################################
Class: configssRNNRBM - Basic setting of the RNN-ssRBM models. 
#########################################################################"""
class configssRNNRBM(configRNNRBM):
    W_Norm = True
    alphaTrain = True
    muTrain = True
    phiTrain = True

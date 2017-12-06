# TODO:
"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the file contains the utility of RBM including Gibbs sampling,
              annealed importance sampling.
              ----2017.11.03
#########################################################################"""

class configRNNRBM:
    aisRun = 100        # <scalar> the number of samples of AIS.
    aisLevel = 10       # <scalar> the number of intermediate proposal distributions of AIS.
    Gibbs = 15          # <scalar> the steps of Gibbs sampling.
    recType = 'LSTM'
    mlpType = 'relu'
    dimMlp = []
    dimRec = []
    dimInput = 100      # <scalar> the size of frame of the input.
    dimState = 100      # <scalar> the size of the stochastic layer.
    init_scale = 0.01   # <scalar> the initialized scales of the weight.
    float = 'float32'   # <string> the type of float.
    Opt = 'SGD'         # <string> the optimization method.
    savePath = None     # <string/None> the path to save the model.
    eventPath = None    # <string/None> the path to save the events for visualization.
    loadPath = None     # <string/None> the path to load the model.



class configssRNNRBM(configRNNRBM):
    W_Norm = False
    alphaTrain = True
    muTrain = True
    phiTrain = True

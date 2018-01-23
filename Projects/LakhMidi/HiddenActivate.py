"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: visualize the hidden activation of the models.
              ----2017.12.21
#########################################################################"""

from Projects.LakhMidi.fetchData import fetchData
from dl4s import binCGRNN, binSRNN, binVRNN, binssRNNRBM
from dl4s import configCGRNN, configSRNN, configVRNN, configssRNNRBM
import os
import numpy as np
import matplotlib.pyplot as plt

# CGRNN
configCGRNN.mode = 'full'
configCGRNN.Opt = 'SGD'
configCGRNN.unitType = 'GRU'
configCGRNN.aisLevel = 100
configCGRNN.aisRun = 100
configCGRNN.dimRec = [500]
configCGRNN.dimMlp = [400, 400]
configCGRNN.dimInput = 128
configCGRNN.dimState = 250
configCGRNN.init_scale = 0.01
configCGRNN.Gibbs = 1
configCGRNN.W_Norm = False
configCGRNN.muTrain = True
configCGRNN.alphaTrain = True
configCGRNN.eventPath = './binCGRNN-f-new/'
configCGRNN.savePath = './binCGRNN-f-new/'
configCGRNN.loadPath = os.path.join(configCGRNN.savePath, 'CGRNN-f')
CGRNN = binCGRNN(configCGRNN)

# SRNN
configSRNN.Opt = 'SGD'
configSRNN.unitType = 'GRU'
configSRNN.mode = 'smooth'
configSRNN.dimRecD = [500]
configSRNN.dimRecA = [500]
configSRNN.dimEnc = [400]
configSRNN.dimDec = [400]
configSRNN.dimMLPx = [400]
configSRNN.dimInput = 128
configSRNN.dimState = 500
configSRNN.init_scale = 0.01
configSRNN.eventPath = './binSRNN-s/'
configSRNN.savePath = './binSRNN-s/'
configSRNN.loadPath = os.path.join(configSRNN.savePath, 'SRNN-s')
SRNN = binSRNN(configSRNN)

# VRNN
configVRNN.Opt = 'SGD'
configVRNN.unitType = 'GRU'
configVRNN.dimRec = [500]
configVRNN.dimForX = [400]
configVRNN.dimForZ = [400]
dimForEnc = [400]
dimForDec = [400]
configVRNN.dimInput = 128
configVRNN.dimState = 500
configVRNN.init_scale = 0.01
configVRNN.eventPath = './binVRNN/'
configVRNN.savePath = './binVRNN/'
configVRNN.loadPath = os.path.join(configVRNN.savePath, 'VRNN-I')
VRNN = binVRNN(configVRNN)

# ssRNN-RBM
configssRNNRBM.Opt = 'SGD'
configssRNNRBM.unitType = 'GRU'
configssRNNRBM.aisLevel = 1
configssRNNRBM.aisRun = 1
configssRNNRBM.dimRec = [500]
configssRNNRBM.dimMlp = [400, 400]
configssRNNRBM.dimInput = 128
configssRNNRBM.dimState = 250
configssRNNRBM.init_scale = 0.01
configssRNNRBM.Gibbs = 1
configssRNNRBM.W_Norm = False
configssRNNRBM.muTrain = True
configssRNNRBM.alphaTrain = True
configssRNNRBM.eventPath = './binssRNNRBM/'
configssRNNRBM.savePath = './binssRNNRBM/'
configssRNNRBM.loadPath = os.path.join(configssRNNRBM.savePath, 'ssRNNRBM')
ssRnnRbm = binssRNNRBM(configssRNNRBM)

# dataset.
Dataset = fetchData()
testSet = Dataset['test']
for i in range(len(testSet)):
    print("[%d/%d]" % (i, len(testSet)))
    x = np.reshape(testSet[i], (1, 240, 128))  # get the batch of input sequence.
    hiddenCGRNN = CGRNN.hidden_function(x)
    hiddenSRNN = SRNN.encoder(x)[0]
    hiddenVRNN = VRNN.encoder(x)[0]
    hiddenssRNNRBM = ssRnnRbm.hidden_function(x)
    # vis.
    plt.figure(1)
    plt.imshow(hiddenCGRNN[0].T, cmap='jet')
    plt.figure(2)
    plt.imshow(hiddenSRNN[0].T, cmap='jet')
    plt.figure(3)
    plt.imshow(hiddenVRNN[0].T, cmap='jet')
    plt.figure(4)
    plt.imshow(hiddenssRNNRBM[0].T, cmap='jet')
    plt.show()

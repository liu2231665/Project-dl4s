"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: generate midi music by the models.
              ----2017.12.27
#########################################################################"""

from Projects.AudioEffects.fetchData import fetchData
from dl4s import gaussCGRNN, gaussSRNN, gaussVRNN, ssRNNRBM
from dl4s import configCGRNN, configSRNN, configVRNN, configssRNNRBM
import os
import numpy as np
import matplotlib.pyplot as plt

# CGRNN
configCGRNN.mode = 'full'
configCGRNN.Opt = 'SGD'
#configCGRNN.recType = 'GRU'
configCGRNN.aisLevel = 100
configCGRNN.aisRun = 100
configCGRNN.dimRec = [500]
configCGRNN.dimMlp = [400, 400]
configCGRNN.dimInput = 150
configCGRNN.dimState = 250
configCGRNN.init_scale = 0.01
configCGRNN.Gibbs = 1
configCGRNN.W_Norm = False
configCGRNN.muTrain = True
configCGRNN.alphaTrain = True
configCGRNN.phiTrain = False
configCGRNN.eventPath = './audioCGRNN-f-new2/'
configCGRNN.savePath = './audioCGRNN-f-new2/'
configCGRNN.loadPath = os.path.join(configCGRNN.savePath, 'CGRNN-f')
CGRNN = gaussCGRNN(configCGRNN)

# SRNN
configSRNN.Opt = 'SGD'
configSRNN.unitType = 'GRU'
configSRNN.mode = 'smooth'
configSRNN.dimRecD = [500]
configSRNN.dimRecA = [500]
configSRNN.dimEnc = [400]
configSRNN.dimDec = [400]
configSRNN.dimMLPx = [400]
configSRNN.dimInput = 150
configSRNN.dimState = 500
configSRNN.init_scale = 0.01
configSRNN.eventPath = './audioSRNN-s/'
configSRNN.savePath = './audioSRNN-s/'
#configSRNN.loadPath = os.path.join(configSRNN.savePath, 'SRNN-s')
SRNN = gaussSRNN(configSRNN)

# VRNN
configVRNN.Opt = 'SGD'
configVRNN.unitType = 'GRU'
configVRNN.dimRec = [500]
configVRNN.dimForX = [400]
configVRNN.dimForZ = [400]
dimForEnc = [400]
dimForDec = [400]
configVRNN.dimInput = 150
configVRNN.dimState = 500
configVRNN.init_scale = 0.01
configVRNN.eventPath = './audioVRNN/'
configVRNN.savePath = './audioVRNN/'
#configVRNN.loadPath = os.path.join(configVRNN.savePath, 'VRNN-I')
VRNN = gaussVRNN(configVRNN)

# ssRNN-RBM
configssRNNRBM.Opt = 'SGD'
configssRNNRBM.unitType = 'GRU'
configssRNNRBM.aisLevel = 1
configssRNNRBM.aisRun = 1
configssRNNRBM.dimRec = [500]
configssRNNRBM.dimMlp = [400, 400]
configssRNNRBM.dimInput = 150
configssRNNRBM.dimState = 250
configssRNNRBM.init_scale = 0.01
configssRNNRBM.Gibbs = 1
configssRNNRBM.W_Norm = False
configssRNNRBM.muTrain = True
configssRNNRBM.alphaTrain = True
configssRNNRBM.eventPath = './audiossRNNRBM/'
configssRNNRBM.savePath = './audiossRNNRBM/'
#configssRNNRBM.loadPath = os.path.join(configssRNNRBM.savePath, 'ssRNNRBM')
ssRnnRbm = ssRNNRBM(configssRNNRBM)

#
CGRNN_FOLDER = "Samples/CGRNN/"
SRNN_FOLDER = "Samples/SRNN/"
VRNN_FOLDER = "Samples/VRNN/"
ssRnnRbm_FOLDER = "Samples/ssRnnRbm/"

#
# dataset.
Dataset = fetchData()
testSet = Dataset['test']
for i in range(20):
    CGRNN_sample = CGRNN.gen_function(testSet[i: i+2])
    SRNN_sample = SRNN.gen_function(100)
    VRNN_sample = VRNN.gen_function(100)
    ssRnnRbm_sample = ssRnnRbm.gen_function(testSet[i: i+2])
    plt.figure(1)
    plt.imshow(testSet[i].T, cmap='hot')
    plt.figure(2)
    plt.imshow(CGRNN_sample[0].T, cmap='hot')
    plt.figure(3)
    plt.imshow(VRNN_sample.T, cmap='hot')
    plt.show()
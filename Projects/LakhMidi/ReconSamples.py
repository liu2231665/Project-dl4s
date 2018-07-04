"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: generate midi music by the models.
              ----2017.12.27
#########################################################################"""

from Projects.LakhMidi.fetchData import fetchData
from dl4s import binCGRNN, binSRNN, binVRNN, binssRNNRBM
from dl4s import configCGRNN, configSRNN, configVRNN, configssRNNRBM
import os
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt

# CGRNN
configCGRNN.mode = 'full'
configCGRNN.Opt = 'SGD'
configCGRNN.recType = 'GRU'
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
configCGRNN.eventPath = './binCGRNN-f-new2/'
configCGRNN.savePath = './binCGRNN-f-new2/'
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

#
CGRNN_FOLDER = "Samples/CGRNN/"
SRNN_FOLDER = "Samples/SRNN/"
VRNN_FOLDER = "Samples/VRNN/"
ssRnnRbm_FOLDER = "Samples/ssRnnRbm/"
Ground_FOLDER = "Samples/"
# Check whether the target path exists.
if not os.path.exists(Ground_FOLDER):
    os.makedirs(Ground_FOLDER)
if not os.path.exists(CGRNN_FOLDER):
    os.makedirs(CGRNN_FOLDER)
if not os.path.exists(SRNN_FOLDER):
    os.makedirs(SRNN_FOLDER)
if not os.path.exists(VRNN_FOLDER):
    os.makedirs(VRNN_FOLDER)
if not os.path.exists(ssRnnRbm_FOLDER):
    os.makedirs(ssRnnRbm_FOLDER)

#
# dataset.
Dataset = fetchData()
testSet = Dataset['test']
for i in range(20):
    print('The ' + str(i) + '-th graph.')
    CGRNN_sample = CGRNN.gen_function(x=testSet[i: i+2])
    SRNN_sample = SRNN.output_function(input=testSet[i: i+2])
    VRNN_sample = VRNN.output_function(input=testSet[i: i+2])
    ssRnnRbm_sample = ssRnnRbm.gen_function(x=testSet[i: i+2])
    plt.figure(1)
    plt.imshow(testSet[i].T, cmap='binary')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(Ground_FOLDER + 'Ground-True-' + str(i) + '.eps')
    plt.clf()
    np.save(Ground_FOLDER + 'Ground-True-' + str(i) + '.npy', testSet[i])
    plt.figure(2)
    plt.imshow(CGRNN_sample[0].T, cmap='binary')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(CGRNN_FOLDER + 'CGRNN-' + str(i) + '.eps')
    plt.clf()
    np.save(CGRNN_FOLDER + 'CGRNN-' + str(i) + '.npy', CGRNN_sample[0])
    plt.figure(3)
    plt.imshow(SRNN_sample[0].T, cmap='binary')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(SRNN_FOLDER + 'SRNN-' + str(i) + '.eps')
    plt.clf()
    np.save(SRNN_FOLDER + 'SRNN-' + str(i) + '.npy', SRNN_sample[0])
    plt.figure(4)
    plt.imshow(VRNN_sample[0].T, cmap='binary')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(VRNN_FOLDER + 'VRNN-' + str(i) + '.eps')
    plt.clf()
    np.save(VRNN_FOLDER + 'VRNN-' + str(i) + '.npy', VRNN_sample[0])
    plt.figure(5)
    plt.imshow(ssRnnRbm_sample[0].T, cmap='binary')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(ssRnnRbm_FOLDER + 'ssRnnRbm-' + str(i) + '.eps')
    plt.clf()
    np.save(ssRnnRbm_FOLDER + 'ssRnnRbm-' + str(i) + '.npy', ssRnnRbm_sample[0])
    plt.close()
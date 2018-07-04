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
import librosa

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
for i in range(10):
    print('The ' + str(i) + '-th graph.')
    CGRNN_sample = CGRNN.gen_function(x=testSet[i: i + 2])
    SRNN_sample = SRNN.output_function(input=testSet[i: i + 2])
    VRNN_sample = VRNN.output_function(input=testSet[i: i + 2])
    ssRnnRbm_sample = ssRnnRbm.gen_function(x=testSet[i: i + 2])
    # reshape them.
    length = np.shape(CGRNN_sample[0])
    length = length[0] * length[1]
    ground = 1.8750e-4 + 0.0413 * np.reshape(testSet[i], (length, ))
    CGRNN_sample = 1.8750e-4 + 0.0413 * np.reshape(CGRNN_sample[0], (length, ))
    SRNN_sample = 1.8750e-4 + 0.0413 * np.reshape(SRNN_sample[0], (length,))
    VRNN_sample = 1.8750e-4 + 0.0413 * np.reshape(VRNN_sample[0], (length,))
    ssRnnRbm_sample = 1.8750e-4 + 0.0413 * np.reshape(ssRnnRbm_sample[0], (length,))
    plt.figure(1)
    plt.plot(ground)
    plt.xticks([])
    plt.savefig(Ground_FOLDER + 'Ground-True-' + str(i) + '.eps')
    plt.clf()
    librosa.output.write_wav(Ground_FOLDER + 'Ground-True-' + str(i) + '.wav', ground, sr=11025)
    plt.figure(2)
    plt.plot(CGRNN_sample)
    plt.xticks([])
    plt.savefig(CGRNN_FOLDER + 'CGRNN-' + str(i) + '.eps')
    plt.clf()
    librosa.output.write_wav(CGRNN_FOLDER + 'CGRNN-' + str(i) + '.wav', CGRNN_sample, sr=11025)
    plt.figure(3)
    plt.plot(SRNN_sample)
    plt.xticks([])
    plt.savefig(SRNN_FOLDER + 'SRNN-' + str(i) + '.eps')
    plt.clf()
    librosa.output.write_wav(SRNN_FOLDER + 'SRNN-' + str(i) + '.wav', SRNN_sample, sr=11025)
    plt.figure(4)
    plt.plot(VRNN_sample)
    plt.xticks([])
    plt.savefig(VRNN_FOLDER + 'VRNN-' + str(i) + '.eps')
    plt.clf()
    librosa.output.write_wav(VRNN_FOLDER + 'VRNN-' + str(i) + '.wav', VRNN_sample, sr=11025)
    plt.figure(5)
    plt.plot(ssRnnRbm_sample)
    plt.xticks([])
    plt.savefig(ssRnnRbm_FOLDER + 'ssRnnRbm-' + str(i) + '.eps')
    plt.clf()
    librosa.output.write_wav(ssRnnRbm_FOLDER + 'ssRnnRbm-' + str(i) + '.wav', ssRnnRbm_sample, sr=11025)
    plt.close()
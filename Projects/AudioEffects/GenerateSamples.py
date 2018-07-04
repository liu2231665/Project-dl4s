"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: generate midi music by the models.
              ----2017.12.29
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
CGRNN_FOLDER = "Generate/CGRNN/"
ssRnnRbm_FOLDER = "Generate/ssRnnRbm/"
# Check whether the target path exists.
if not os.path.exists(CGRNN_FOLDER):
    os.makedirs(CGRNN_FOLDER)
if not os.path.exists(ssRnnRbm_FOLDER):
    os.makedirs(ssRnnRbm_FOLDER)

#
for i in range(1):
    print('The ' + str(i) + '-th graph.')
    CGRNN_sample = CGRNN.gen_function(numSteps=150, gibbs=100)
    # ssRnnRbm_sample = ssRnnRbm.gen_function(numSteps=150, gibbs=100)
    #
    plt.figure(1)
    plt.imshow(CGRNN_sample.T, cmap='binary')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(CGRNN_FOLDER + 'CGRNN-' + str(i) + '.eps')
    plt.clf()
    np.save(CGRNN_FOLDER + 'CGRNN-' + str(i) + '.npy', CGRNN_sample)
    #
    # plt.figure(2)
    # plt.imshow(ssRnnRbm_sample.T, cmap='binary')
    # plt.xticks([])
    # plt.yticks([])
    # plt.savefig(ssRnnRbm_FOLDER + 'ssRnnRbm-' + str(i) + '.eps')
    # plt.clf()
    # np.save(ssRnnRbm_FOLDER + 'ssRnnRbm-' + str(i) + '.npy', ssRnnRbm_sample)
    plt.close()
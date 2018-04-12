"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: display the covariance and precision by the models.
              ----2017.12.30
#########################################################################"""

from Projects.AudioEffects.fetchData import fetchData
from dl4s import gaussCGRNN, gaussSRNN, gaussVRNN, ssRNNRBM
from dl4s import configCGRNN, configSRNN, configVRNN, configssRNNRBM
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

SMALL_SIZE = 22
matplotlib.rc('font', size=SMALL_SIZE)

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
CGRNN_FOLDER = "COVPRE/CGRNN/"
ssRnnRbm_FOLDER = "COVPRE/ssRnnRbm/"
# Check whether the target path exists.
if not os.path.exists(CGRNN_FOLDER):
    os.makedirs(CGRNN_FOLDER)
if not os.path.exists(ssRnnRbm_FOLDER):
    os.makedirs(ssRnnRbm_FOLDER)

#
# dataset.
Dataset = fetchData()
testSet = Dataset['test']
for i in range(1):
    i = i + 1000
    print('The ' + str(i) + '-th graph.')
    CGRNN_pre = CGRNN.pre_function(input=testSet[i: i + 2])[0]
    CGRNN_cov = CGRNN.cov_function(input=testSet[i: i + 2])[0]
    ssRNNRBM_pre = ssRnnRbm.pre_function(input=testSet[i: i + 2])[0]
    ssRNNRBM_cov = ssRnnRbm.cov_function(input=testSet[i: i + 2])[0]

    #
    plt.figure(1)
    plt.imshow(ssRNNRBM_pre[0], cmap='Blues')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('dim of Vn')
    plt.ylabel('dim of Vn')
    plt.title('n = 0')
    plt.savefig(ssRnnRbm_FOLDER + 'ssRnnRbm-pre-' + str(i) + '- 0.eps')
    plt.figure(2)
    plt.imshow(ssRNNRBM_pre[60], cmap='Blues')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('dim of Vn')
    plt.ylabel('dim of Vn')
    plt.title('n = 60')
    plt.savefig(ssRnnRbm_FOLDER + 'ssRnnRbm-pre-' + str(i) + '- 60.eps')
    plt.figure(3)
    im = plt.imshow(ssRNNRBM_pre[140], cmap='Blues')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('dim of Vn')
    plt.ylabel('dim of Vn')
    plt.title('n = 140')
    plt.colorbar(mappable=im, orientation="vertical", ticks=[])
    plt.savefig(ssRnnRbm_FOLDER + 'ssRnnRbm-pre-' + str(i) + '- 140.eps')

    #
    plt.figure(4)
    plt.imshow(CGRNN_pre[0], cmap='Blues')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('dim of Vn')
    plt.ylabel('dim of Vn')
    plt.title('n = 0')
    plt.savefig(CGRNN_FOLDER + 'CGRNN-pre-' + str(i) + '- 0.eps')
    plt.figure(5)
    plt.imshow(CGRNN_pre[60], cmap='Blues')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('dim of Vn')
    plt.ylabel('dim of Vn')
    plt.title('n = 60')
    plt.savefig(CGRNN_FOLDER + 'CGRNN-pre-' + str(i) + '- 60.eps')
    plt.figure(6)
    im = plt.imshow(CGRNN_pre[140], cmap='Blues')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('dim of Vn')
    plt.ylabel('dim of Vn')
    plt.title('n = 140')
    plt.colorbar(mappable=im, orientation="vertical", ticks=[])
    plt.savefig(CGRNN_FOLDER + 'CGRNN-pre-' + str(i) + '- 140.eps')

    #
    plt.figure(7)
    plt.imshow(CGRNN_cov[0], cmap='Blues')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('dim of Vn')
    plt.ylabel('dim of Vn')
    plt.title('n = 0')
    plt.savefig(CGRNN_FOLDER + 'CGRNN-cov-' + str(i) + '- 0.eps')
    plt.figure(8)
    plt.imshow(CGRNN_cov[60], cmap='Blues')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('dim of Vn')
    plt.ylabel('dim of Vn')
    plt.title('n = 60')
    plt.savefig(CGRNN_FOLDER + 'CGRNN-cov-' + str(i) + '- 60.eps')
    plt.figure(9)
    im = plt.imshow(CGRNN_cov[140], cmap='Blues')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('dim of Vn')
    plt.ylabel('dim of Vn')
    plt.title('n = 140')
    plt.colorbar(mappable=im, orientation="vertical", ticks=[])
    plt.savefig(CGRNN_FOLDER + 'CGRNN-cov-' + str(i) + '- 140.eps')
    plt.show()
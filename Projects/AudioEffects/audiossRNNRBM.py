"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the code to train and run the ssRNNRBM
              under the "Audio Effect".
              ----2017.12.06
#########################################################################"""
from dl4s import ssRNNRBM, gaussSRNN
from dl4s import configssRNNRBM as Config
from dl4s import configSRNN
from Projects.AudioEffects.fetchData import fetchData
from Projects.AudioEffects.rmseTool import rmseGaussRNNRBM
import os

Config.Opt = 'Adam'
Config.unitType = 'GRU'
Config.aisLevel = 30
Config.aisRun = 20
Config.dimRec = [500]
Config.dimMlp = [400, 400]
Config.dimInput = 150
Config.dimState = 250
Config.init_scale = 0.01
Config.Gibbs = 1
Config.W_Norm = True
Config.muTrain = True
Config.alphaTrain = True
Config.eventPath = './audiossRNNRBM/'
Config.savePath = './audiossRNNRBM/'
SAVETO = './audiossRNNRBM/historyaudio_ssRNNRBM.npz'

Flag = 'evaluation'                       # {'training'/'evaluation'}

if __name__ == '__main__':
    if Flag == 'training':
        Dataset = fetchData()
        # Check whether the target event path exists.
        if not os.path.exists(Config.eventPath):
            os.makedirs(Config.eventPath)
        # Check whether the target saving path exists.
        if not os.path.exists(Config.savePath):
            os.makedirs(Config.savePath)
        RnnRbm = ssRNNRBM(Config, Bound=(-25.0, 25.0))
        RnnRbm.full_train(dataset=Dataset, maxEpoch=300, batchSize=125, earlyStop=300, learning_rate=0.001,
                       valid_batchSize=125, saveto=SAVETO)

    if Flag == 'evaluation':
        Dataset = fetchData()
        configSRNN.Opt = 'Adam'
        configSRNN.unitType = 'GRU'
        configSRNN.mode = 'filter'
        configSRNN.dimRecD = [500]
        configSRNN.dimRecA = [500]
        configSRNN.dimEnc = [400]
        configSRNN.dimDec = [400]
        configSRNN.dimMLPx = [400]
        configSRNN.dimInput = 150
        configSRNN.dimState = 500
        configSRNN.init_scale = 0.01
        configSRNN.eventPath = './audioSRNN-f/'
        configSRNN.savePath = './audioSRNN-f/'
        SAVETO = './audioSRNN-f/historyaudioSRNN-f.npz'
        configSRNN.loadPath = configSRNN.savePath
        SRNN = gaussSRNN(configSRNN)
        Config.aisRun = 100
        Config.loadPath = Config.savePath
        RnnRbm = ssRNNRBM(Config, Bound=(-25.0, 25.0), VAE=SRNN)
        print('Evaluation: start computing the RMSE metric.')
        RMSE, NLL = rmseGaussRNNRBM(RnnRbm, Dataset['test'], batchSize=25)
        print('The testing reconstructed error is \x1b[1;91m%10.4f\x1b[0m.' % RMSE)
        print('The testing reconstructed NLL is \x1b[1;91m%10.4f\x1b[0m.' % NLL)


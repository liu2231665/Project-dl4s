"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the code to train and run the RNNRBM
              under the "Audio Effect".
              ----2017.12.01
#########################################################################"""
from dl4s import gaussRnnRBM, gaussSRNN
from dl4s import configRNNRBM as Config
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
Config.dimState = 500
Config.init_scale = 0.0001
Config.Gibbs = 1
Config.eventPath = './audioRNNRBM/'
Config.savePath = './audioRNNRBM/'
SAVETO = './audioRNNRBM/historyaudioRNNRBM.npz'

Flag = 'evaluation'                       # {'training'/'evaluation'}

if __name__ == '__main__':
    Dataset = fetchData()

    if Flag == 'training':
        # Check whether the target event path exists.
        if not os.path.exists(Config.eventPath):
            os.makedirs(Config.eventPath)
        # Check whether the target saving path exists.
        if not os.path.exists(Config.savePath):
            os.makedirs(Config.savePath)
        RnnRbm = gaussRnnRBM(Config)
        RnnRbm.full_train(dataset=Dataset, maxEpoch=300, batchSize=125, earlyStop=10, learning_rate=0.001,
                       valid_batchSize=125, saveto=SAVETO)

    if Flag == 'evaluation':
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
        RnnRbm = gaussRnnRBM(Config, VAE=SRNN)
        print('Evaluation: start computing the RMSE metric.')
        RMSE, NLL = rmseGaussRNNRBM(RnnRbm, Dataset['test'], batchSize=25)
        print('The testing reconstructed error is \x1b[1;91m%10.4f\x1b[0m.' % RMSE)
        print('The testing reconstructed NLL is \x1b[1;91m%10.4f\x1b[0m.' % NLL)


"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the code to train and run the SRNN
              under the "Audio Effect".
              ----2017.11.28
#########################################################################"""
from dl4s import gaussSRNN
from dl4s import configSRNN as Config
from Projects.AudioEffects.fetchData import fetchData
from Projects.AudioEffects.rmseTool import rmseRNN
import os

Config.Opt = 'Adam'
Config.unitType = 'GRU'
Config.mode = 'filter'
Config.dimRecD = [500]
Config.dimRecA = [500]
Config.dimEnc = [400]
Config.dimDec = [400]
Config.dimMLPx = [400]
Config.dimInput = 150
Config.dimState = 500
Config.init_scale = 0.01
Config.eventPath = './audioSRNN-f/'
Config.savePath = './audioSRNN-f/'
SAVETO = './audioSRNN-f/historyaudioSRNN-f.npz'

Flag = 'training'                       # {'training'/'evaluation'}

if __name__ == '__main__':
    Dataset = fetchData()

    if Flag == 'training':
        # Check whether the target event path exists.
        if not os.path.exists(Config.eventPath):
            os.makedirs(Config.eventPath)
        # Check whether the target saving path exists.
        if not os.path.exists(Config.savePath):
            os.makedirs(Config.savePath)
        SRNN = gaussSRNN(Config)
        # SRNN.full_train(dataset=Dataset, maxEpoch=280, batchSize=125, earlyStop=10, learning_rate=0.00001, for smooth
        #                valid_batchSize=125, saveto=SAVETO)
        SRNN.full_train(dataset=Dataset, maxEpoch=300, batchSize=125, earlyStop=10, learning_rate=0.0001,
                        valid_batchSize=125, saveto=SAVETO)

    if Flag == 'evaluation':
        Config.loadPath = Config.savePath
        SRNN = gaussSRNN(Config)
        print('Evaluation: start computing the RMSE metric.')
        RMSE = rmseRNN(SRNN, Dataset['test'], batchSize=125)
        print('The testing reconstructed error is \x1b[1;91m%10.4f\x1b[0m.' % RMSE)


"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the code to train and run the VRNN
              under the "Audio Effect".
              ----2017.11.03
#########################################################################"""
from dl4s import gaussVRNN
from dl4s import configVRNN as Config
from Projects.AudioEffects.fetchData import fetchData
from Projects.AudioEffects.rmseTool import rmseRNN
import os

Config.Opt = 'Adam'
Config.unitType = 'GRU'
Config.dimRec = [500]
Config.dimForX = [400]
Config.dimForZ = [400]
dimForEnc = [400]
dimForDec = [400]
Config.dimInput = 150
Config.dimState = 500
Config.init_scale = 0.01
Config.eventPath = './audioVRNN/'
Config.savePath = './audioVRNN/'
SAVETO = './audioVRNN/historyaudioVRNN.npz'

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
        VRNN = gaussVRNN(Config)
        VRNN.full_train(dataset=Dataset, maxEpoch=300, batchSize=125, earlyStop=10, learning_rate=0.0001,
                       valid_batchSize=125, saveto=SAVETO)

    if Flag == 'evaluation':
        Config.loadPath = Config.savePath
        VRNN = gaussVRNN(Config)
        print('Evaluation: start computing the RMSE metric.')
        RMSE = rmseRNN(VRNN, Dataset['test'], batchSize=125)
        print('The testing reconstructed error is \x1b[1;91m%10.4f\x1b[0m.' % RMSE)


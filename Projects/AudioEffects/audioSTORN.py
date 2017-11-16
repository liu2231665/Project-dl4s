"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the code to train and run the gaussSTORN from dl4s.autoregRNN
              under the "Blizzard 2013 data-set".
              ----2017.11.03
#########################################################################"""
from dl4s import gaussSTORN
from dl4s import configSTORN as Config
from Projects.AudioEffects.fetchData import fetchData
from Projects.AudioEffects.rmseTool import rmseRNN
import os
import h5py

Config.Opt = 'Adam'
Config.unitType = 'GRU'
Config.dimGen = [500]
Config.dimReg = [500]
Config.dimInput = 150
Config.dimState = 500
Config.init_scale = 0.01
Config.savePath = "./audioSTORN/"
Config.eventPath = "./audioSTORN/"
SAVETO = './audioSTORN/historySTORN.npz'

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
        # Add the save file name into the save path.
        Config.savePath = os.path.join(Config.savePath, 'STORN')
        RNN = gaussSTORN(Config)
        RNN.full_train(dataset=Dataset, maxEpoch=300, batchSize=125, earlyStop=10, learning_rate=0.001,
                       valid_batchSize=125, saveto=SAVETO)

    if Flag == 'evaluation':
        Config.loadPath = os.path.join(Config.savePath, 'STORN')
        RNN = gaussSTORN(Config)
        print('Evaluation: start computing the RMSE metric.')
        RMSE = rmseRNN(RNN, Dataset['test'], batchSize=125)
        print('The testing reconstructed error is \x1b[1;91m%10.4f\x1b[0m.' % RMSE)


#TODO:
"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the code to train and run the gaussRNN from dl4s.autoregRNN
              under the "Blizzard 2013 data-set".
              ----2017.11.03
#########################################################################"""
from dl4s import gaussRNN, configRNN
from fetchData import fetchData
import os
import h5py

configRNN.unitType = "GRU"
configRNN.Opt = 'Adam'
configRNN.savePath = "./blizzardRNN/"
configRNN.eventPath = "./blizzardRNN/"
configRNN.dimLayer = [200, 1024, 200]
configRNN.init_scale = 0.01
SAVETO = './blizzardRNN/historyblizzardRNN.npz'

Flag = 'training'                       # {'training'/'evaluation'}

if __name__ == '__main__':
    Dataset = fetchData()

    if Flag == 'training':
        # Check whether the target event path exists.
        if not os.path.exists(configRNN.eventPath):
            os.makedirs(configRNN.eventPath)
        # Check whether the target saving path exists.
        if not os.path.exists(configRNN.savePath):
            os.makedirs(configRNN.savePath)
        # Add the save file name into the save path.
        configRNN.savePath = os.path.join(configRNN.savePath, 'RNN-I')
        RNN = gaussRNN(configRNN)
        RNN.full_train(dataset=Dataset, maxEpoch=50, batchSize= 125, earlyStop=300, learning_rate=0.001,
                       valid_batchSize=125, saveto=SAVETO)

    if Flag == 'evaluation':
        pass

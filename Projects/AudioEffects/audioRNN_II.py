"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the code to train and run the RNN-II
              under the "Audio Effect".
              ----2017.11.03
#########################################################################"""
from dl4s import gaussRNN, configRNN
from dl4s import full_train
from Projects.AudioEffects.fetchData import fetchData
from Projects.AudioEffects.rmseTool import rmseRNN
import os

configRNN.unitType = "GRU"
configRNN.Opt = 'Adam'
configRNN.savePath = "./audioRNN-II/"
configRNN.eventPath = "./audioRNN-II/"
configRNN.dimLayer = [150, 500, 500, 150]
configRNN.init_scale = 0.01
SAVETO = './audioRNN-II/historyRNN.npz'

Flag = 'evaluation'                       # {'training'/'evaluation'}

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
        configRNN.savePath = os.path.join(configRNN.savePath, 'RNN-II')
        RNN = gaussRNN(configRNN)
        full_train(model=RNN, dataset=Dataset, maxEpoch=300, batchSize= 125, earlyStop=10, learning_rate=0.001,
                       valid_batchSize=125, saveto=SAVETO)

    if Flag == 'evaluation':
        configRNN.loadPath = os.path.join(configRNN.savePath, 'RNN-II')
        RNN = gaussRNN(configRNN)
        print('Evaluation: start computing the RMSE metric.')
        RMSE = rmseRNN(RNN, Dataset['test'], batchSize=125)
        print('The testing reconstructed error is \x1b[1;91m%10.4f\x1b[0m.' % RMSE)

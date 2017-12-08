"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the code to train and run the ssRNNRBM
              under the "Audio Effect".
              ----2017.12.06
#########################################################################"""
from dl4s import ssRNNRBM
from dl4s import configssRNNRBM as Config
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
Config.init_scale = 0.01
Config.Gibbs = 1
Config.W_Norm = True
Config.muTrain = True
Config.alphaTrain = True
Config.eventPath = './audiossRNNRBM/'
Config.savePath = './audiossRNNRBM/'
SAVETO = './audiossRNNRBM/historyaudio_ssRNNRBM.npz'

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
        RnnRbm = ssRNNRBM(Config, Bound=(-20.0, 20.0))
        RnnRbm.full_train(dataset=Dataset, maxEpoch=300, batchSize=125, earlyStop=10, learning_rate=0.00,
                       valid_batchSize=125, saveto=SAVETO)

    if Flag == 'evaluation':
        Config.aisLevel = 1000
        Config.aisRun = 20
        Config.loadPath = Config.savePath
        RnnRbm = ssRNNRBM(Config, Bound=(-20.0, 20.0))
        print('Evaluation: start computing the RMSE metric.')
        RMSE, NLL = rmseGaussRNNRBM(RnnRbm, Dataset['test'], batchSize=125)
        print('The testing reconstructed error is \x1b[1;91m%10.4f\x1b[0m.' % RMSE)
        print('The testing reconstructed NLL is \x1b[1;91m%10.4f\x1b[0m.' % NLL)


"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the code to train and run the gaussian CGRNN from dl4s.CGRNN
              under the "Audio Effect data-set".
              ----2017.12.27
#########################################################################"""
from Projects.AudioEffects.fetchData import fetchData
from dl4s import gaussCGRNN, gaussSRNN
from dl4s import configCGRNN as Config
from dl4s import configSRNN
from dl4s.tools import full_train
from Projects.AudioEffects.rmseTool import rmseGaussRNNRBM
import os

Config.mode = 'full'
Config.Opt = 'Adam'
Config.unitType = 'GRU'
Config.aisLevel = 100
Config.aisRun = 100
Config.dimRec = [200]
Config.dimMlp = []
Config.dimInput = 150
Config.dimState = 75
Config.init_scale = 0.01
Config.Gibbs = 1
Config.W_Norm = True
Config.muTrain = True
Config.alphaTrain = True
Config.eventPath = './audioCGRNN-f-5/'
Config.savePath = './audioCGRNN-f-5/'
SAVETO = './audioCGRNN-f-5/historyCGRNN-f.npz'

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
        Config.savePath = os.path.join(Config.savePath, 'CGRNN-f')
        # Build the model and prepare the data-set.
        RnnRbm = gaussCGRNN(Config)
        full_train(model=RnnRbm, dataset=Dataset, maxEpoch=300, batchSize=125, earlyStop=300, learning_rate=0.0005,
                          valid_batchSize=75, saveto=SAVETO)

    if Flag == 'evaluation':
        # REFERENCE MODEL.
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
        #
        Config.loadPath = os.path.join(Config.savePath, 'CGRNN-s')
        RnnRbm = gaussCGRNN(Config, VAE=SRNN)
        print('Evaluation: start computing the accuracy metric.')
        ACC, NLL = rmseGaussRNNRBM(RnnRbm, Dataset['test'], batchSize=25)
        print('The testing transcription accuracy is \x1b[1;91m%10.4f\x1b[0m.' % ACC)
        print('The testing transcription NLL is \x1b[1;91m%10.4f\x1b[0m.' % NLL)

"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the code to train and run the binary ssRNNRBM from dl4s.TRBM
              under the "Lakh Midi data-set".
              ----2017.12.16
#########################################################################"""
from Projects.LakhMidi.fetchData import fetchData
from dl4s import binssRNNRBM, binSRNN
from dl4s import configssRNNRBM as Config
from dl4s import configSRNN
from Projects.LakhMidi.accTool import accRBM
import os

Config.Opt = 'SGD'
Config.unitType = 'GRU'
Config.aisLevel = 5000
Config.aisRun = 50
Config.dimRec = [500]
Config.dimMlp = [400, 400]
Config.dimInput = 128
Config.dimState = 250
Config.init_scale = 0.01
Config.Gibbs = 1
Config.W_Norm = False
Config.muTrain = True
Config.alphaTrain = True
Config.eventPath = './binssRNNRBM/'
Config.savePath = './binssRNNRBM/'
SAVETO = './binssRNNRBM/historyMidiRNNRBM.npz'

Flag = 'evaluation'                       # {'training'/'evaluation'}

if __name__ == '__main__':
    # REFERENCE MODEL.
    Dataset = fetchData()
    configSRNN.Opt = 'SGD'
    configSRNN.unitType = 'GRU'
    configSRNN.mode = 'smooth'
    configSRNN.dimRecD = [500]
    configSRNN.dimRecA = [500]
    configSRNN.dimEnc = [400]
    configSRNN.dimDec = [400]
    configSRNN.dimMLPx = [400]
    configSRNN.dimInput = 128
    configSRNN.dimState = 500
    configSRNN.init_scale = 0.01
    configSRNN.eventPath = './binSRNN-s/'
    configSRNN.savePath = './binSRNN-s/'
    configSRNN.loadPath = os.path.join(configSRNN.savePath, 'SRNN-s')
    SRNN = binSRNN(configSRNN)

    if Flag == 'training':
        # Check whether the target event path exists.
        if not os.path.exists(Config.eventPath):
            os.makedirs(Config.eventPath)
        # Check whether the target saving path exists.
        if not os.path.exists(Config.savePath):
            os.makedirs(Config.savePath)
        # Add the save file name into the save path.
        Config.savePath = os.path.join(Config.savePath, 'ssRNNRBM')
        # Build the model and prepare the data-set.
        RnnRbm = binssRNNRBM(Config)
        RnnRbm.full_train(dataset=Dataset, maxEpoch=300, batchSize=125, earlyStop=300, learning_rate=0.1,
                          valid_batchSize=25, saveto=SAVETO)

    if Flag == 'evaluation':
        Config.loadPath = os.path.join(Config.savePath, 'ssRNNRBM')
        RnnRbm = binssRNNRBM(Config)
        print('Evaluation: start computing the accuracy metric.')
        ACC, NLL = accRBM(RnnRbm, Dataset['test'], batchSize=25)
        print('The testing transcription accuracy is \x1b[1;91m%10.4f\x1b[0m.' % ACC)
        print('The testing transcription NLL is \x1b[1;91m%10.4f\x1b[0m.' % NLL)

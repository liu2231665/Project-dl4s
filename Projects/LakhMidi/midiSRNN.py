"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the code to train and run the binSRNN from dl4s.seqVAE
              under the "Lakh Midi data-set".
              ----2017.11.18
#########################################################################"""
from Projects.LakhMidi.fetchData import fetchData
from dl4s import binSRNN
from dl4s import configSRNN as Config
from Projects.LakhMidi.accTool import accRNN
import os

Config.Opt = 'SGD'
Config.unitType = 'GRU'
Config.mode = 'filter'
Config.dimRecD = [500]
Config.dimRecA = [500]
Config.dimEnc = [400]
Config.dimDec = [400]
Config.dimMLPx = [400]
Config.dimInput = 128
Config.dimState = 500
Config.init_scale = 0.01
Config.eventPath = './binSRNN-f/'
Config.savePath = './binSRNN-f/'
SAVETO = './binSRNN-f/historyMidiSRNN-f.npz'

Flag = 'training'                       # {'training'/'evaluation'}

if __name__ == '__main__':
    if Flag == 'training':
        # Check whether the target event path exists.
        if not os.path.exists(Config.eventPath):
            os.makedirs(Config.eventPath)
        # Check whether the target saving path exists.
        if not os.path.exists(Config.savePath):
            os.makedirs(Config.savePath)
        # Add the save file name into the save path.
        Config.savePath = os.path.join(Config.savePath, 'SRNN-f')
        # Build the model and prepare the data-set.
        Dataset = fetchData()
        SRNN = binSRNN(Config)
        SRNN.full_train(dataset=Dataset, maxEpoch=300, batchSize=125, earlyStop=10, learning_rate=0.05,
                          valid_batchSize=125, saveto=SAVETO)

    if Flag == 'evaluation':
        Config.loadPath = os.path.join(Config.savePath, 'SRNN-s')
        Dataset = fetchData()
        SRNN = binSRNN(Config)
        print('Evaluation: start computing the accuracy metric.')
        ACC = accRNN(SRNN, Dataset['test'], batchSize=125)
        print('The testing transcription accuracy is \x1b[1;91m%10.4f\x1b[0m.' % ACC)

"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the code to train and run the binRNNRBM from dl4s.TRBM
              under the "Lakh Midi data-set".
              ----2017.11.23
#########################################################################"""
from Projects.LakhMidi.fetchData import fetchData
from dl4s import binRnnRBM
from dl4s import configRNNRBM as Config
from Projects.LakhMidi.accTool import accRBM
import os

Config.Opt = 'SGD'
Config.unitType = 'GRU'
Config.aisLevel = 5000
Config.aisRun = 20
Config.dimRec = [500]
Config.dimMlp = [400, 400]
Config.dimInput = 128
Config.dimState = 500
Config.init_scale = 0.01
Config.Gibbs = 1
Config.eventPath = './binRNNRBM/'
Config.savePath = './binRNNRBM/'
SAVETO = './binRNNRBM/historyMidiRNNRBM.npz'

Flag = 'evaluation'                       # {'training'/'evaluation'}

if __name__ == '__main__':
    if Flag == 'training':
        # Check whether the target event path exists.
        if not os.path.exists(Config.eventPath):
            os.makedirs(Config.eventPath)
        # Check whether the target saving path exists.
        if not os.path.exists(Config.savePath):
            os.makedirs(Config.savePath)
        # Add the save file name into the save path.
        Config.savePath = os.path.join(Config.savePath, 'RNNRBM')
        # Build the model and prepare the data-set.
        Dataset = fetchData()
        RnnRbm = binRnnRBM(Config)
        RnnRbm.full_train(dataset=Dataset, maxEpoch=300, batchSize=125, earlyStop=10, learning_rate=0.1,
                          valid_batchSize=25, saveto=SAVETO)

    if Flag == 'evaluation':
        Config.loadPath = os.path.join(Config.savePath, 'RNNRBM')
        Dataset = fetchData()
        RnnRbm = binRnnRBM(Config)
        print('Evaluation: start computing the accuracy metric.')
        ACC, NLL = accRBM(RnnRbm, Dataset['test'], batchSize=25)
        print('The testing transcription accuracy is \x1b[1;91m%10.4f\x1b[0m.' % ACC)
        print('The testing transcription NLL is \x1b[1;91m%10.4f\x1b[0m.' % NLL)

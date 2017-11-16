"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the code to train and run the binSTORN from dl4s.seqVAE
              under the "Lakh Midi data-set".
              ----2017.11.11
#########################################################################"""
from Projects.LakhMidi.fetchData import fetchData
from dl4s import binSTORN
from dl4s import configSTORN as Config
from Projects.LakhMidi.accTool import accRNN
import os

Config.Opt = 'SGD'
Config.unitType = 'GRU'
Config.dimGen = [500]
Config.dimReg = [500]
Config.dimInput = 128
Config.dimState = 500
Config.init_scale = 0.01
Config.eventPath = './binSTORN/'
Config.savePath = './binSTORN/'
SAVETO = './binSTORN/historyMidiSTORN.npz'

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
        Config.savePath = os.path.join(Config.savePath, 'STORN-I')
        # Build the model and prepare the data-set.
        Dataset = fetchData()
        STORN = binSTORN(Config)
        STORN.full_train(dataset=Dataset, maxEpoch=300, batchSize=125, earlyStop=10, learning_rate=0.03,
                          valid_batchSize=125, saveto=SAVETO)

    if Flag == 'evaluation':
        Config.loadPath = os.path.join(Config.savePath, 'STORN-II')
        Dataset = fetchData()
        STORN = binSTORN(Config)
        print('Evaluation: start computing the accuracy metric.')
        ACC = accRNN(STORN, Dataset['test'], batchSize=125)

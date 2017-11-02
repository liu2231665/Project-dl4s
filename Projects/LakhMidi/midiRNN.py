"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the code to train and run the binRNN from dl4s.autoregRNN
              under the "Lakh Midi data-set".
              ----2017.11.01
#########################################################################"""
from Projects.LakhMidi.fetchData import fetchData
from dl4s import binRNN, configRNN
import os

configRNN.unitType = "GRU"
configRNN.savePath = "./binRNN/"
configRNN.eventPath = "./binRNN/"
configRNN.dimLayer = [128, 500, 128]
configRNN.init_scale = 0.01
SAVETO = './binRNN/historyMidiRNN.npz'

Flag = 'training'                       # {'training'/'evaluation'}

if __name__ == '__main__':
    if Flag == 'training':
        # Check whether the target event path exists.
        if not os.path.exists(configRNN.eventPath):
            os.makedirs(configRNN.eventPath)
        # Check whether the target saving path exists.
        if not os.path.exists(configRNN.savePath):
            os.makedirs(configRNN.savePath)
        # Add the save file name into the save path.
        configRNN.savePath = os.path.join(configRNN.savePath, 'RNN-I')
        # Build the model and prepare the data-set.
        Dataset = fetchData()
        RNN = binRNN(configRNN)
        RNN.full_train(dataset=Dataset, maxEpoch=500, batchSize=125, earlyStop=10, learning_rate=0.1,
                          valid_batchSize=125, saveto=SAVETO)

    if Flag == 'evaluation':
        pass
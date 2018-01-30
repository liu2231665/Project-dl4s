"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the code to train and run the binRNN from dl4s.autoregRNN
              under the "Lakh Midi data-set".
              ----2017.11.01
#########################################################################"""
from Projects.LakhMidi.fetchData import fetchData
from dl4s import binRNN, configRNN
from dl4s import full_train
from Projects.LakhMidi.accTool import accRNN
import os
import matplotlib.pyplot as plt

configRNN.unitType = "GRU"
#configRNN.Opt = 'Momentum'
configRNN.savePath = "./binRNN/"
configRNN.eventPath = "./binRNN/"
configRNN.dimLayer = [128, 500, 128]
configRNN.init_scale = 0.01
SAVETO = './binRNN/historyMidiRNN.npz'

Flag = 'evaluation'                       # {'training'/'evaluation'}

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
        full_train(model=RNN, dataset=Dataset, maxEpoch=300, batchSize=125, earlyStop=10, learning_rate=0.1,
                          valid_batchSize=125, saveto=SAVETO)

    if Flag == 'evaluation':
        configRNN.loadPath = os.path.join(configRNN.savePath, 'RNN-I')
        Dataset = fetchData()
        RNN = binRNN(configRNN)
        print('Evaluation: start computing the accuracy metric.')
        ACC = accRNN(RNN, Dataset['test'], batchSize=125)
        print('The testing transcription accuracy is \x1b[1;91m%10.4f\x1b[0m.' % ACC)

"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the code to train and run the binVRNN from dl4s.seqVAE
              under the "Lakh Midi data-set".
              ----2017.11.15
#########################################################################"""
from Projects.LakhMidi.fetchData import fetchData
from dl4s import binVRNN
from dl4s import full_train
from dl4s import configVRNN as Config
from Projects.LakhMidi.accTool import accRNN
import os
import matplotlib.pyplot as plt

Config.Opt = 'SGD'
Config.unitType = 'GRU'
Config.dimRec = [500]
Config.dimForX = [400]
Config.dimForZ = [400]
dimForEnc = [400]
dimForDec = [400]
Config.dimInput = 128
Config.dimState = 500
Config.init_scale = 0.01
Config.eventPath = './binVRNN/'
Config.savePath = './binVRNN/'
SAVETO = './binVRNN/historyMidiVRNN.npz'

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
        Config.savePath = os.path.join(Config.savePath, 'VRNN-I')
        # Build the model and prepare the data-set.
        Dataset = fetchData()
        VRNN = binVRNN(Config)
        full_train(model=VRNN, dataset=Dataset, maxEpoch=300, batchSize=125, earlyStop=10, learning_rate=0.03,
                          valid_batchSize=125, saveto=SAVETO)

    if Flag == 'evaluation':
        Config.loadPath = os.path.join(Config.savePath, 'VRNN-I')
        Dataset = fetchData()
        VRNN = binVRNN(Config)
        Midi = VRNN.gen_function(numSteps=100)
        plt.figure(1)
        plt.imshow(Midi.T, cmap='binary')
        plt.xticks([])
        plt.yticks([])
        plt.show()
        print('Evaluation: start computing the accuracy metric.')
        ACC = accRNN(VRNN, Dataset['test'], batchSize=125)
        print('The testing transcription accuracy is \x1b[1;91m%10.4f\x1b[0m.' % ACC)

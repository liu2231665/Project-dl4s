"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: transer the numpy files of the midi songs into midi files.
              (Cause the code privided by RNN-RBM tutorial to save midi
              runs in python 2.7 but my code is in python 3.6)
              ----2017.12.29
#########################################################################"""
import numpy as np
from midi.utils import midiread, midiwrite
#
CGRNN_FOLDER = "Samples/CGRNN/"
SRNN_FOLDER = "Samples/SRNN/"
VRNN_FOLDER = "Samples/VRNN/"
ssRnnRbm_FOLDER = "Samples/ssRnnRbm/"
Ground_FOLDER = "Samples/"

for i in range(20):
    print('The ' + str(i) + '-th graph.')
    Ground_sample = np.load(Ground_FOLDER + 'Ground-True-' + str(i) + '.npy')
    CGRNN_sample = np.load(CGRNN_FOLDER + 'CGRNN-' + str(i) + '.npy')
    SRNN_sample = np.load(SRNN_FOLDER + 'SRNN-' + str(i) + '.npy')
    VRNN_sample = np.load(VRNN_FOLDER + 'VRNN-' + str(i) + '.npy')
    ssRnnRbm_sample = np.load(ssRnnRbm_FOLDER + 'ssRnnRbm-' + str(i) + '.npy')
    midiwrite(Ground_FOLDER + 'Ground-True-' + str(i) + '.mid', Ground_sample, (1, 128), 0.25)
    midiwrite(CGRNN_FOLDER + 'CGRNN-' + str(i) + '.mid', CGRNN_sample, (1, 128), 0.25)
    midiwrite(SRNN_FOLDER + 'SRNN-' + str(i) + '.mid', SRNN_sample, (1, 128), 0.25)
    midiwrite(VRNN_FOLDER + 'VRNN-' + str(i) + '.mid', VRNN_sample, (1, 128), 0.25)
    midiwrite(ssRnnRbm_FOLDER + 'ssRnnRbm-' + str(i) + '.mid', ssRnnRbm_sample, (1, 128), 0.25)
    pass
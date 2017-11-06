"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the files contain the tools to pre-process the IDMT-SMT-
              Audio-Effects data-set. The dataset can be downloaded in
              <https://www.idmt.fraunhofer.de/en/business_units/m2d/smt/
              audio_effects.html>.
              Note: In our projects, we use only the raw .wav files and no
              features are considered. Hence, we rearrange the directories
              as MAIN_CAT/SUB_CAT. Please merge each document-2 into document
              -1 and move the folders inside the Samples one layer above. Then
              rename the main documents as I done.
              ----2017.11.01
#########################################################################"""
import os, h5py
import librosa, librosa.display
import numpy as np
from dl4s.tools import get_batches_idx

TRAIN_RATIO = 0.9
Valid_RATIO = 0.95

AE_HDF5 = "./dataset/AudioEffects.hdf5"
# The main categories devided by instruments.
MAIN_CAT = ["./dataset/bass_mono",
            "./dataset/gitar_mono",
            "./dataset/gitar_poly"
            ]
# The sub categories devided by effects.
SUB_CAT = ["Chorus", "Distortion", "EQ", "FeedbackDelay", "Flanger", "NoFX",
           "Overdrive", "Phaser", "Reverb", "SlapbackDelay", "Tremolo", "Vibrato"
           ]

"""#########################################################################
findWAV: return the list of wav files in the directory and subdirectories 
          of path.
input: path - the root path.
output: Dir - the list of wav files.
#########################################################################"""
def findWAV(PATH):
    Dir = []
    for dirName, subdirList, fileList in os.walk(PATH):
        for name in fileList:
            midiPath = os.path.join(dirName, name)
            # Check whether the path is a midi file.
            if midiPath[-4:] == '.wav':
                Dir.append(midiPath)
    return Dir

"""#########################################################################
fetchData: normalize the original dataset by the global mean and std of the
           the data-set and return it.
output: None.
output: Dataset - the normalized dataset.
#########################################################################"""
def fetchData():
    Dataset = None
    times = 1
    while 1:
        if os.path.exists(AE_HDF5):
            # load the .hdf5 dataset
            print("\x1b[1;34m----->> LOAD THE DATASET <<-----\x1b[0m")
            Dataset = h5py.File(AE_HDF5, 'r')
            break
        elif all(os.path.exists(path) for path in MAIN_CAT):
            print("Step \x1b[1;34m%d\x1b[0m: process the raw dataset." % times)
            times += 1
            with h5py.File(AE_HDF5, 'w') as Dataset:
                Dataset.create_dataset('train', (1, 147, 150), maxshape=(None, 147, 150), chunks=True)
                Dataset.create_dataset('valid', (1, 147, 150), maxshape=(None, 147, 150), chunks=True)
                Dataset.create_dataset('test', (1, 147, 150), maxshape=(None, 147, 150), chunks=True)
                trainEND = 0
                validEND = 0
                testEND = 0

                #
                for mainCat in MAIN_CAT:
                    for subCat in SUB_CAT:
                        PATH = os.path.join(mainCat, subCat)
                        print("\x1b[1;36m%s:\x1b[0m" % PATH)
                        Dir = findWAV(PATH)                         # get the file paths in the direction.
                        L = len(Dir)
                        idx = 1
                        for wav in Dir:
                            print("\x1b[1;35m%d/%d\x1b[0m: \x1b[1;34m%s\x1b[0m" % (idx, L, wav))
                            idx += 1
                            waveform = librosa.load(wav, sr=22050/2)[0]
                            waveform = waveform[0:22050].reshape(147, 150)
                            rand = np.random.uniform(0, 1.0)
                            # Split into train/valid/test sets.
                            if rand < TRAIN_RATIO:
                                # save to train.
                                Dataset['train'].resize((trainEND + 1, 147, 150))
                                Dataset['train'][trainEND:trainEND + 1, :, :] = waveform
                                trainEND += 1
                                pass
                            elif rand < Valid_RATIO:
                                # save to valid.
                                Dataset['valid'].resize((validEND + 1, 147, 150))
                                Dataset['valid'][validEND:validEND + 1, :, :] = waveform
                                validEND += 1
                                pass
                            else:
                                # save to test.
                                Dataset['test'].resize((testEND + 1, 147, 150))
                                Dataset['test'][testEND:testEND + 1, :, :] = waveform
                                testEND += 1
                # normalize the data-set.
                print("Step \x1b[1;34m%d\x1b[0m: normalize the data-set." % times)
                times += 1
                batches = get_batches_idx(len(Dataset['train']), batch_size=1024, shuffle=False)
                # compute the mean
                temp = []
                batchLen = []
                for idx in batches:
                    x = Dataset['train'][idx.tolist()]
                    temp.append(x.mean())
                    batchLen.append(x.shape[0])
                mean = np.asarray(temp) * np.asarray(batchLen)
                mean = mean.sum() / len(Dataset['train'])
                # comute the std.
                temp = []
                for idx in batches:
                    x = Dataset['train'][idx.tolist()]
                    temp1 = (x - mean) ** 2
                    temp.append(temp1.mean())
                std = np.asarray(temp) * np.asarray(batchLen)
                std = np.sqrt(std.sum() / len(Dataset['train']))
                print("Acess the mean \x1b[1;36m%10.4f\x1b[0m and standard deviation  "
                      "\x1b[1;36m%10.4f\x1b[0m." % (mean, std))
                # save the normalized dataset.
                Dataset.create_dataset('mean', data=mean)
                Dataset.create_dataset('std', data=std)
                print("Step \x1b[1;34m%d\x1b[0m: save the normalized dataset." % times)
                times += 1
                # save the train set.
                for idx in batches:
                    x = Dataset['train'][idx.tolist()]
                    Dataset['train'][idx.tolist()] = (x - mean) / std
                # save the valid set.
                batches = get_batches_idx(len(Dataset['valid']), batch_size=1024, shuffle=False)
                for idx in batches:
                    x = Dataset['valid'][idx.tolist()]
                    Dataset['valid'][idx.tolist()] = (x - mean) / std
                # save the valid set.
                batches = get_batches_idx(len(Dataset['test']), batch_size=1024, shuffle=False)
                for idx in batches:
                    x = Dataset['test'][idx.tolist()]
                    Dataset['test'][idx.tolist()] = (x - mean) / std
        else:
            raise (ValueError("Either the processed data-set or the raw data is needed!!"))

    if times > 4:
        raise (ValueError("The data fetching process is out of control!!"))

    if Dataset is None:
        raise (ValueError("The dataset is not loaded properly!!"))
    return Dataset


if __name__ == '__main__':
    Dataset = fetchData()
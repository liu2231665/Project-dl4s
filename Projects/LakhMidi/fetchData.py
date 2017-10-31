import os
import urllib.request
import tarfile
import h5py
import pretty_midi
import numpy as np

# Data name.URL.
Lakh_HDF5 = "./dataset/Lakh_clean.hdf5"
Lakh_RAW = "./dataset/clean_midi.tar.gz"
Lakh_URL = "http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz"

TRAIN_RATIO = 0.8
Valid_RATIO = 0.9

"""
listFile: return the list of midi files in the directory and subdirectories of path.
input: path - the root path.
output: Dir - the list of midi files.
"""
def listFile(path):
    Dir = []
    for dirName, subdirList, fileList in os.walk(path):
        for name in fileList:
            midiPath = os.path.join(dirName, name)
            if midiPath[-4:] == '.mid':
                Dir.append(midiPath)
    return Dir

"""
readMIDI: read the midi file and transfer it into piano-rolls.
input: path - the path of the midi file.
output: midi - the binary numpy array represents the piano-rolls 
        (for the convenience of my research, I remove the information of the instruments).
"""
def readMIDI(path):
    midi = pretty_midi.PrettyMIDI(path).get_piano_roll(fs=4).T
    midi = midi > 0
    return np.asarray(midi, 'float32')


def fetchData():
    Dataset = None
    times = 1
    while 1:
        if os.path.exists(Lakh_HDF5):
            # TODO: load the .hdf5 dataset
            print("\x1b[1;34m----->> LOAD THE DATASET <<-----\x1b[0m")
            break
        elif os.path.exists(Lakh_RAW):
            # TODO: prerpocess the raw data and save as .hdf5
            print("Step \x1b[1;34m%d\x1b[0m: unzip the raw data." % times)
            times +=1
            # unzip the files.
            tar = tarfile.open(Lakh_RAW, "r:gz")
            tar.extractall(path='./dataset/')
            tar.close()
            Dir = listFile('./dataset/')
            print("Step \x1b[1;34m%d\x1b[0m: preprocess the raw data." % times)
            # read the raw data and save into hdf5.
            with h5py.File(Lakh_HDF5, 'w') as Dataset:
                Dataset.create_dataset('train', (1, 240, 128), maxshape=(None, 240, 128), chunks=True)
                Dataset.create_dataset('valid', (1, 240, 128), maxshape=(None, 240, 128), chunks=True)
                Dataset.create_dataset('test', (1, 240, 128), maxshape=(None, 240, 128), chunks=True)

                trainEND = 0
                validEND = 0
                testEND = 0

                for midiPath in Dir:
                    midi = readMIDI(midiPath)
                    start = 0
                    while start + 240 <= midi.shape[0]:
                        rand = np.random.uniform(0, 1.0001)
                        if rand < TRAIN_RATIO:
                            # save to train.
                            Dataset['train'].resize((trainEND+1, 240, 128))
                            Dataset['train'][trainEND:trainEND+1, :, :] = np.reshape(midi[start:start+240, :],
                                                                                     [1, 240, 128])
                            trainEND += 1
                            pass
                        elif rand < Valid_RATIO:
                            # save to valid.
                            Dataset['valid'].resize((validEND + 1, 240, 128))
                            Dataset['valid'][validEND:validEND + 1, :, :] = np.reshape(midi[start:start + 240, :],
                                                                                       [1, 240, 128])
                            validEND += 1
                            pass
                        else:
                            # save to test.
                            Dataset['test'].resize((testEND + 1, 240, 128))
                            Dataset['test'][testEND:testEND + 1, :, :] = np.reshape(midi[start:start + 240, :],
                                                                                        [1, 240, 128])
                            testEND += 1
                        #
                        start += 240


        else:
            # download the raw data.
            print("Step \x1b[1;34m%d\x1b[0m: download the raw data." % times)
            times += 1
            if not os.path.exists('./dataset'):             # make a new folder to save the data.
                os.makedirs('./dataset')
            urllib.request.urlretrieve(Lakh_URL, './dataset/clean_midi.tar.gz')
            if not os.path.exists('./dataset'):
                raise(ValueError("The lakh_clean dataset is not downloaded properly!!"))

        if times > 3:
            raise(ValueError("The data fetching process is out of control!!"))

    if Dataset is None:
            raise(ValueError("The dataset is not loaded properly!!"))
    return Dataset

if __name__ == '__main__':
    #fetchData()
    pass
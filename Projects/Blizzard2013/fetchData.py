"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the files contain the tools to return the normalized data-set.
              ----2017.11.01
#########################################################################"""
import os, h5py
import numpy as np
from dl4s.tools import get_batches_idx

Blizzard_HDF5 = "./dataset/blizzard.hdf5"
Blizzard_normalize_HDF5 = "./dataset/blizzard_n.hdf5"

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
        if os.path.exists(Blizzard_normalize_HDF5):
            # load the .hdf5 dataset
            print("\x1b[1;34m----->> LOAD THE DATASET <<-----\x1b[0m")
            Dataset = h5py.File(Blizzard_normalize_HDF5, 'r')
            break
        elif os.path.exists(Blizzard_HDF5):
            print("Step \x1b[1;34m%d\x1b[0m: normalize the raw dataset." % times)
            times += 1
            Dataset = h5py.File(Blizzard_HDF5, 'r')
            #L = int(len(Dataset['train']) / 2)
            train = Dataset['train']
            #L = int(len(Dataset['valid']) / 2)
            valid = Dataset['valid']
            #L = int(len(Dataset['test']) / 2)
            test = Dataset['test']
            batches = get_batches_idx(len(train), batch_size=1024, shuffle=False)
            # compute the mean
            temp = []
            batchLen = []
            for idx in batches:
                x = train[idx.tolist()]
                temp.append(x.mean())
                batchLen.append(x.shape[0])
            mean = np.asarray(temp) * np.asarray(batchLen)
            mean = mean.sum() / len(train)
            # comute the std.
            temp = []
            for idx in batches:
                x = train[idx.tolist()]
                temp1 = (x - mean) ** 2
                temp.append(temp1.mean())
            std = np.asarray(temp) * np.asarray(batchLen)
            std = np.sqrt(std.sum() / len(train))
            print("Acess the mean \x1b[1;91m%10.4f\x1b[0m and standard deviation  \x1b[1;91m%10.4f\x1b[0m." % (mean, std))

            print("Step \x1b[1;34m%d\x1b[0m: save the normalized dataset." % times)
            times += 1
            with h5py.File(Blizzard_normalize_HDF5, 'w') as Dataset_n:
                Dataset_n.create_dataset('train', (1, 40, 200), maxshape=(None, 40, 200), chunks=True)
                Dataset_n.create_dataset('valid', (1, 40, 200), maxshape=(None, 40, 200), chunks=True)
                Dataset_n.create_dataset('test', (1, 40, 200), maxshape=(None, 40, 200), chunks=True)
                Dataset_n.create_dataset('mean', data=mean)
                Dataset_n.create_dataset('std', data=std)
                # save the train set.
                End = 0
                for idx in batches:
                    x = train[idx.tolist()]
                    Dataset_n['train'].resize((End + len(idx), 40, 200))
                    Dataset_n['train'][End:End + len(idx), :, :] = (x - mean) / std
                    End += len(idx)

                # save the valid set.
                End = 0
                batches = get_batches_idx(len(valid), batch_size=1024, shuffle=False)
                for idx in batches:
                    x = valid[idx.tolist()]
                    Dataset_n['valid'].resize((End + len(idx), 40, 200))
                    Dataset_n['valid'][End:End + len(idx), :, :] = (x - mean) / std
                    End += len(idx)

                # save the test set.
                End = 0
                batches = get_batches_idx(len(test), batch_size=1024, shuffle=False)
                for idx in batches:
                    x = test[idx.tolist()]
                    Dataset_n['test'].resize((End + len(idx), 40, 200))
                    Dataset_n['test'][End:End + len(idx), :, :] = (x - mean) / std
                    End += len(idx)


            print("Step \x1b[1;34m%d\x1b[0m: remove the raw dataset." % times)
            times += 1
            #os.remove(Blizzard_HDF5)
        else:
            raise (ValueError("The blizzard dataset is needed!!"))

    if times > 4:
        raise (ValueError("The data fetching process is out of control!!"))

    if Dataset is None:
        raise (ValueError("The dataset is not loaded properly!!"))
    return Dataset
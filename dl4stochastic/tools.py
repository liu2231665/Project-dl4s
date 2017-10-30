# Common Tools such as dataset fetching.
# Author: Yingru Liu
# Institute: Stony Brook University

import numpy as np

"""
get_minibatches_idx: Used to shuffle the dataset at each iteration.
input: len - the length the dataset section.
       batch_size - the batch size.
       shuffle - bool indicating whether shuffle the idx.
"""
def get_batches_idx(len, batch_size, shuffle=True):

    idx_list = np.arange(len, dtype="int32")
    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(len // batch_size):
        batch = idx_list[minibatch_start:minibatch_start + batch_size]
        batch = np.sort(batch)
        minibatches.append(batch)
        minibatch_start += batch_size

    if (minibatch_start != len):
        # Make a minibatch out of what is left
        batch = idx_list[minibatch_start:]
        batch = np.sort(batch)
        minibatches.append(batch)

    return minibatches
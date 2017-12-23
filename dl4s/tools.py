"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: Common Tools that will be used by all the models.
              ----2017.11.01
#########################################################################"""

import numpy as np
import tensorflow as tf
import time

"""#########################MATH TOOLS##############################"""
"""#########################MATH TOOLS##############################"""
"""#########################MATH TOOLS##############################"""

"""#########################################################################
BernoulliNLL: function to compute the negative log-likelihood of Bernoulli
              distribution.
input: x - network input indicated by <tensor placeholder>. 
       P - the probability of 1.
output: nll - a tensor representing the NLL per bit.
#########################################################################"""
def BernoulliNLL(x, P):
    nll = x * tf.log(P+1e-8) + (1 - x) * tf.log(1-P+1e-8)
    return -tf.reduce_mean(nll)

"""#########################################################################
GaussNLL: function to compute the negative log-likelihood of Gaussian 
          distribution with a diagonal covariance matrix.
input: x - network input indicated by <tensor placeholder>. 
       mean - mean of the Gaussian distribution computed by the graph.
       sigma - variance of the Gaussian distribution computed by the graph.
output: nll - a tensor representing the NLL per bit.
#########################################################################"""
def GaussNLL(x, mean, sigma):
    nll = 0.5*tf.reduce_mean(tf.div(tf.square(x-mean), sigma) + tf.log(sigma)) + 0.5*tf.log(2*np.pi)
    return nll

"""#########################################################################
GaussKL: function to compute KL divergence of  two Gaussian distributions
         with a diagonal covariance matrices.
input: meanP - mean of the Gaussian distribution "P"
       sigmaP - variance of the Gaussian distribution "P".
       meanQ - mean of the Gaussian distribution "P"
       sigmaQ - variance of the Gaussian distribution "P".
output: kl - a tensor representing the KL divergence per bit.
#########################################################################"""
def GaussKL(meanP, sigmaP, meanQ, sigmaQ):
    term1 = tf.log(sigmaQ + 1e-8) - tf.log(sigmaP + 1e-8)
    term2 = tf.div(sigmaP + (meanP - meanQ)**2, sigmaQ + 1e-8)
    return 0.5 * tf.reduce_mean(term1 + term2) - 0.5


"""#########################MODEL TRAINING##############################"""
"""#########################MODEL TRAINING##############################"""
"""#########################MODEL TRAINING##############################"""

"""#########################################################################
get_minibatches_idx: Used to shuffle the dataset at each iteration.
input: len - the length the dataset section.
       batch_size - the batch size.
       shuffle - bool indicating whether shuffle the idx.
#########################################################################"""
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

"""#########################################################################
saveModel:save the trained model into disk.
input: model - the model.
       savePath - another saving path, if not provide, use the default path.
output: None
#########################################################################"""
def saveModel(model, savePath=None):
    with model._graph.as_default():
        # Create a saver.
        saver = tf.train.Saver()
        if savePath is None:
            saver.save(model._sess, model._savePath)
        else:
            saver.save(model._sess, savePath)
    return

"""#########################################################################
loadModel:load the model from disk.
input: model - the model.
       loadPath - another loading path, if not provide, use the default path.
output: None
#########################################################################"""
def loadModel(model, loadPath=None):
    with model._graph.as_default():
        # Create a saver.
        saver = tf.train.Saver()
        if loadPath is None:
            if model._loadPath is not None:
                saver.restore(model._sess, model._loadPath)
            else:
                raise (ValueError("No loadPath is given!"))
        else:
            saver.restore(model._sess, loadPath)
    return


"""#########################################################################
saveEvent:save the event to visualize the last model once.
          (To visualize other aspects, other codes should be used.)
input: model - the model.
       eventPath - another loading path, if not provide, use the default path.
output: None
#########################################################################"""
def saveEvent(model, eventPath=None):
    if eventPath is None:
        if model._eventPath is None:
            raise ValueError("Please privide the path to save the events by self._eventPath!!")
        else:
            eventPath = model._eventPath
    with model._graph.as_default():
        # compute the statistics of the parameters.
        for param in model._params:
            scopeName =param.name.split('/')[-1]
            with tf.variable_scope(scopeName[0:-2]):
                mean = tf.reduce_mean(param)
                tf.summary.scalar('mean',  mean)
                stddev = tf.sqrt(tf.reduce_mean(tf.square(param - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(param))
                tf.summary.scalar('min', tf.reduce_min(param))
                tf.summary.histogram('histogram', param)
        # visualize the trainable parameters of the model.
        summary = tf.summary.merge_all()
        summary_str = model._sess.run(summary)
        # Define a event writer and write the events into disk.
        Writer = tf.summary.FileWriter(eventPath, model._sess.graph)
        Writer.add_summary(summary_str)
        Writer.flush()
        Writer.close()
    return

"""#########################################################################
full_train: define to fully train a model given the dataset.
input: model - the model.
       dataset - the dataset used to train. The split should be train/valid
                 /test.
       maxEpoch - the maximum epoches to train the model.
       batchSize - the batch size for training.
       earlyStop - the tolerance epoches for early stopping.
       learning_rate - you know what it is...
       saveto - the additional save path that may be different from
                the default to save the history loss during training.
       valid_batchSize - the batch size for validation and testing.
output: None.
#########################################################################"""
def full_train(model, dataset, maxEpoch, batchSize, earlyStop,
               learning_rate, saveto, valid_batchSize=1):
    # slit the dataset.
    trainData = dataset['train']
    validData = dataset['valid']
    testData = dataset['test']
    validBatch = get_batches_idx(len(validData), valid_batchSize, False)
    testBatch = get_batches_idx(len(testData), valid_batchSize, False)

    historyLoss = []                   # <list> record the training process.
    durations = []                      # <list> record the training duration.
    worseCase = 0                       # indicate the worse cases for early stopping
    bestEpoch = -1

    # Issue: the idx should be a list. Hence, .tolist() is required.
    for epoch in range(maxEpoch):
        start_time = time.time() # the start time of epoch.
        # update the model w.r.t the training set and record the average loss.
        trainLoss = []
        trainBatch = get_batches_idx(len(trainData), batchSize, True)
        for Idx in trainBatch:
            x = trainData[Idx.tolist()]
            trainLoss.append(x.shape[0]*model.train_function(x, learning_rate))
        trainLoss_avg = np.asarray(trainLoss).sum()/len(trainData)

        duration = time.time() - start_time  # the duration of one epoch.
        durations.append(duration)

        # evaluate the model w.r.t the valid set and record the average loss.
        validLoss = []
        for Idx in validBatch:
            x = validData[Idx.tolist()]
            validLoss.append(x.shape[0]*model.val_function(x))
        validLoss_avg = np.asarray(validLoss).sum()/len(validData)
        print("In epoch \x1b[1;32m%4d\x1b[0m: the training loss is "
              "\x1b[1;32m%10.4f\x1b[0m; the valid loss is \x1b[1;32m%10.4f\x1b[0m." % (epoch, trainLoss_avg, validLoss_avg))

        # check the early stopping conditions.
        if len(historyLoss)==0 or validLoss_avg < np.min(np.asarray(historyLoss)[:, 1]):
            worseCase = 0
            bestEpoch = epoch
            saveModel(model)
        else:
            worseCase += 1
        historyLoss.append([trainLoss_avg, validLoss_avg])
        if worseCase >= earlyStop:
            break

    durationEpoch = np.asarray(durations).mean()
    print('The average epoch duration is \x1b[1;91m%10.4f\x1b[0m seconds.' % durationEpoch)

    # evaluate the best model w.r.t the test set and record the average loss.
    loadModel(model, model._savePath)

    trainLoss = []
    trainBatch = get_batches_idx(len(trainData), batchSize, True)
    for Idx in trainBatch:
        x = trainData[Idx.tolist()]
        trainLoss.append(x.shape[0] * model.val_function(x))
    trainLoss_avg = np.asarray(trainLoss).sum() / len(trainData)

    validLoss = []
    for Idx in validBatch:
        x = validData[Idx.tolist()]
        validLoss.append(x.shape[0] * model.val_function(x))
    validLoss_avg = np.asarray(validLoss).sum() / len(validData)

    testLoss = []
    for Idx in testBatch:
        x = testData[Idx.tolist()]
        testLoss.append(x.shape[0] * model.val_function(x))
    testLoss_avg = np.asarray(testLoss).sum() / len(testData)

    # evaluate the model w.r.t the valid set and record the average loss.
    print("BEST MODEL from epoch \x1b[1;91m%4d\x1b[0m with training loss"
          " \x1b[1;91m%10.4f\x1b[0m and valid loss \x1b[1;91m%10.4f\x1b[0m."
          % (bestEpoch, trainLoss_avg, validLoss_avg))
    print('The testing loss is \x1b[1;91m%10.4f\x1b[0m.' % testLoss_avg)

    if saveto is not None:
        np.savez(saveto, historyLoss=np.asarray(historyLoss),
                 testLoss=testLoss_avg, durationEpoch=durationEpoch)
    return

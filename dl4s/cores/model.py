"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: This file contains the abstract of configuration and model for
              all models in this package.
#########################################################################"""
import numpy as np
import tensorflow as tf
import time

"""#########################################################################
Class: _config - the hyper abstraction of model configuration.
#########################################################################"""
class _config(object):
    init_scale = 0.1            # <scalar> the initialized scales of the weight.
    float = 'float32'           # <string> the type of float.
    Opt = 'SGD'                 # <string> the optimization method.
    savePath = None             # <string/None> the path to save the model.
    eventPath = None            # <string/None> the path to save the events for visualization.
    loadPath = None             # <string/None> the path to load the model.
    dimIN = None                  # <int> dimension of input.

"""#########################################################################
Class: _model - the hyper abstraction of neural models.
#########################################################################"""
class _model(object):
    """#########################################################################
    __init__:the initialization function.
    input: Config - configuration class in ./utility.
    output: None.
    #########################################################################"""
    def __init__(
            self,
            config=_config(),
    ):
        # <tensor graph> define a default graph.
        self._graph = tf.Graph()
        with self._graph.as_default():
            # <tensor placeholder> input.
            self.x = tf.placeholder(dtype='float32', shape=[None, None, config.dimIN])
            # <tensor placeholder> learning rate.
            self.lr = tf.placeholder(dtype='float32', shape=(), name='learningRate')
            # <tensor placeholder> the length of generated samples.
            self.sampleLen = tf.placeholder(dtype='int32', shape=(), name='sampleLen')

            # <string/None> path to save the model.
            self._savePath = config.savePath
            # <string/None> path to save the events.
            self._eventPath = config.eventPath
            # <string/None> path to load the events.
            self._loadPath = config.loadPath
            # <list> collection of trainable parameters.
            self._params = []
            # <pass> will be define in the children classes.
            self._train_step = None
            # <pass> will be define in the children classes.
            self._loss = None
            # <pass> will be define in the children classes.
            self._outputs = None
            # <pass> will be define in the children classes.
            self._gen_operator = None
            # <pass> will be define in the children classes.
            self._feature = None

            # <Tensorflow Optimizer>.
            if config.Opt == 'Adadelta':
                self._optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
            elif config.Opt == 'Adam':
                self._optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            elif config.Opt == 'Momentum':
                self._optimizer = tf.train.MomentumOptimizer(self.lr, 0.9)
            elif config.Opt == 'SGD':
                self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            else:
                raise (ValueError("Config.Opt should be either 'Adadelta', 'Adam', 'Momentum' or 'SGD'!"))

            # <Tensorflow Session>.
            self._sess = tf.Session(graph=self._graph)

    """#########################################################################
    _runSession: initialize the graph or restore from the load path.
    input: None.
    output: None.
    #########################################################################"""
    def _runSession(self):
        if self._loadPath is None:
            self._sess.run(tf.global_variables_initializer())
        else:
            saver = tf.train.Saver()
            saver.restore(self._sess, self._loadPath)
        return

    """#########################################################################
    saveModel:save the trained model into disk.
    input: model - the model.
           savePath - another saving path, if not provide, use the default path.
    output: None
    #########################################################################"""
    def saveModel(self, savePath=None):
        with self._graph.as_default():
            # Create a saver.
            saver = tf.train.Saver()
            if savePath is None:
                if self._savePath is not None:
                    saver.save(self._sess, self._savePath)
            else:
                saver.save(self._sess, savePath)
        return

    """#########################################################################
    loadModel:load the model from disk.
    input: model - the model.
           loadPath - another loading path, if not provide, use the default path.
    output: None
    #########################################################################"""
    def loadModel(self, loadPath=None):
        try:
            with self._graph.as_default():
                # Create a saver.
                saver = tf.train.Saver()
                if loadPath is None:
                    if self._loadPath is not None:
                        saver.restore(self._sess, self._loadPath)
                    else:
                        raise (ValueError("No loadPath is given!"))
                else:
                    saver.restore(self._sess, loadPath)
        except:
            print("\x1b[1;91mNo saved model is given. Initiate the model randomly...\x1b[0m")
        return

    """#########################################################################
    saveEvent:save the event to visualize the last model once.
              (To visualize other aspects, other codes should be used.)
    input: model - the model.
           eventPath - another loading path, if not provide, use the default path.
    output: None
    #########################################################################"""
    def saveEvent(self, eventPath=None):
        try:
            if eventPath is None:
                if self._eventPath is None:
                    raise ValueError("Please privide the path to save the events by self._eventPath!!")
                else:
                    eventPath = self._eventPath
            with self._graph.as_default():
                # compute the statistics of the parameters.
                for param in self._params:
                    scopeName = param.name.split('/')[-1]
                    with tf.variable_scope(scopeName[0:-2]):
                        mean = tf.reduce_mean(param)
                        tf.summary.scalar('mean', mean)
                        stddev = tf.sqrt(tf.reduce_mean(tf.square(param - mean)))
                        tf.summary.scalar('stddev', stddev)
                        tf.summary.scalar('max', tf.reduce_max(param))
                        tf.summary.scalar('min', tf.reduce_min(param))
                        tf.summary.histogram('histogram', param)
                # visualize the trainable parameters of the model.
                summary = tf.summary.merge_all()
                summary_str = self._sess.run(summary)
                # Define a event writer and write the events into disk.
                Writer = tf.summary.FileWriter(eventPath, self._sess.graph)
                Writer.add_summary(summary_str)
                Writer.flush()
                Writer.close()
        except:
            print("\x1b[1;91mNo path is provided to save events. continue the program...\x1b[0m")
        return

    # TODO:
    def impaint(self, input, mask):
        pass

    """#########################################################################
    reconstruct: reconstruction the noise-free version of data.
    #########################################################################"""
    def reconstruct(self, input):
        with self._graph.as_default():
            class_type = self.__class__.__name__
            if class_type == "binRNN" or class_type == "gaussRNN" or \
                class_type == "binSTORN" or class_type == "gaussSTORN":
                zero_padd = np.zeros(shape=(input.shape[0], 1, input.shape[2]), dtype='float32')
                con_input = np.concatenate((zero_padd, input), axis=1)
                output = self._sess.run(self._outputs, feed_dict={self.x: con_input})
                return output[:, 0:-1, :]
            else:
                return self._sess.run(self._outputs, feed_dict={self.x: input})

    """#########################################################################
    embed: return extracted feature/descriptor/representation of data.
    #########################################################################"""
    def embed(self, input, *args, **kwargs):
        # if the input is not in batch format, reshape it.
        if len(input.shape) == 2:
            input = input.reshape([-1, input.shape[0], input.shape[1]])
        elif len(input.shape) == 1:
            input = input.reshape([-1, 1, input.shape[0]])
        #
        with self._graph.as_default():
            class_type = self.__class__.__name__
            # if ssRBM is included in the model. There is an alternative sparse feature.
            if (class_type == "ssRNNRBM" or class_type == "binssRNNRBM") \
                    and ('sparse' in args or 'sparse' in kwargs):
                return self._sess.run(self._sparse_feature, feed_dict={self.x: input})
            else:
                return self._sess.run(self._feature, feed_dict={self.x: input})

    """#########################################################################
    generate: generate sample sequence with the length as numSteps.
    #########################################################################"""
    def generate(self, numSteps):
        with self._graph.as_default():
            return self._sess.run(self._gen_operator, feed_dict={self.sampleLen: numSteps})

    """#########################################################################
    train_function: compute the loss and update the tensor variables.
    input: input - numerical input.
           lrate - <scalar> learning rate.
    output: the loss value.
    #########################################################################"""
    def train_function(self, input, lrate, *args, **kwargs):
        with self._graph.as_default():
            class_type = self.__class__.__name__
            if class_type == "binRNN" or class_type == "gaussRNN" or \
                class_type == "binSTORN" or class_type == "gaussSTORN":
                zero_padd = np.zeros(shape=(input.shape[0], 1, input.shape[2]), dtype='float32')
                input = np.concatenate((zero_padd, input), axis=1)
            #
            _, loss_value = self._sess.run([self._train_step, self._loss],
                                           feed_dict={self.x: input, self.lr: lrate})
        return loss_value * input.shape[-1]

    """#########################################################################
    val_function: compute the loss with given input.
    input: input - numerical input.
    output: the loss value.
    #########################################################################"""
    def val_function(self, input, *args, **kwargs):
        with self._graph.as_default():
            class_type = self.__class__.__name__
            if class_type == "binRNN" or class_type == "gaussRNN" or \
                class_type == "binSTORN" or class_type == "gaussSTORN":
                zero_padd = np.zeros(shape=(input.shape[0], 1, input.shape[2]), dtype='float32')
                input = np.concatenate((zero_padd, input), axis=1)
            loss_value = self._sess.run(self._loss, feed_dict={self.x: input})
        return loss_value * input.shape[-1]

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
    def full_train(self, dataset, maxEpoch, batchSize, earlyStop,
                   learning_rate, saveto, valid_batchSize=1, *args, **kwargs):
        with self._graph.as_default():
            # define the tf.data.Dataset object.
            trainData_placeholder = tf.placeholder(dataset['train'].dtype, dataset['train'].shape)
            trainData = tf.data.Dataset.from_tensor_slices(trainData_placeholder).shuffle(10000)
            trainData = trainData.batch(batchSize).shuffle(10000)
            train_iterator = trainData.make_initializable_iterator()
            next_train_batch = train_iterator.get_next()
            #
            validData_placeholder = tf.placeholder(dataset['valid'].dtype, dataset['valid'].shape)
            validData = tf.data.Dataset.from_tensor_slices(validData_placeholder)
            validData = validData.batch(valid_batchSize)
            valid_iterator = validData.make_initializable_iterator()
            next_valid_batch = valid_iterator.get_next()
            #
            testData_placeholder = tf.placeholder(dataset['test'].dtype, dataset['test'].shape)
            testData = tf.data.Dataset.from_tensor_slices(testData_placeholder)
            testData = testData.batch(valid_batchSize)
            test_iterator = testData.make_initializable_iterator()
            next_test_batch = test_iterator.get_next()

        historyLoss = []  # <list> record the training process.
        durations = []  # <list> record the training duration.
        worseCase = 0  # indicate the worse cases for early stopping
        bestEpoch = -1

        # Issue: the idx should be a list. Hence, .tolist() is required.
        for epoch in range(maxEpoch):
            start_time = time.time()  # the start time of epoch.
            # update the model w.r.t the training set and record the average loss.
            trainLoss = []
            self._sess.run(train_iterator.initializer, feed_dict={trainData_placeholder: dataset['train']})
            while True:
                try:
                    x = self._sess.run(next_train_batch)
                    trainLoss.append(x.shape[0] * self.train_function(x, learning_rate))
                except tf.errors.OutOfRangeError:
                    break
            trainLoss_avg = np.asarray(trainLoss).sum() / len(dataset['train'])

            duration = time.time() - start_time  # the duration of one epoch.
            durations.append(duration)

            # evaluate the model w.r.t the valid set and record the average loss.
            validLoss = []
            self._sess.run(valid_iterator.initializer, feed_dict={validData_placeholder: dataset['valid']})
            while True:
                try:
                    x = self._sess.run(next_valid_batch)
                    validLoss.append(x.shape[0] * self.val_function(x))
                except tf.errors.OutOfRangeError:
                    break
            validLoss_avg = np.asarray(validLoss).sum() / len(dataset['valid'])
            print("In epoch \x1b[1;32m%4d\x1b[0m: the training loss is "
                  "\x1b[1;32m%10.4f\x1b[0m; the valid loss is \x1b[1;32m%10.4f\x1b[0m." % (
                      epoch, trainLoss_avg, validLoss_avg))

            # check the early stopping conditions.
            if len(historyLoss) == 0 or validLoss_avg < np.min(np.asarray(historyLoss)[:, 1]):
                worseCase = 0
                bestEpoch = epoch
                self.saveModel(saveto)
            else:
                worseCase += 1
            historyLoss.append([trainLoss_avg, validLoss_avg])
            if worseCase >= earlyStop:
                break

        durationEpoch = np.asarray(durations).mean()
        print('The average epoch duration is \x1b[1;91m%10.4f\x1b[0m seconds.' % durationEpoch)

        # evaluate the best model w.r.t the test set and record the average loss if
        # we have saved the parameters of th best model.
        if saveto is not None:
            self.loadModel(saveto)

            trainLoss = []
            self._sess.run(train_iterator.initializer, feed_dict={trainData_placeholder: dataset['train']})
            while True:
                try:
                    x = self._sess.run(next_train_batch)
                    trainLoss.append(x.shape[0] * self.val_function(x))
                except tf.errors.OutOfRangeError:
                    break
            trainLoss_avg = np.asarray(trainLoss).sum() / len(dataset['train'])


            validLoss = []
            self._sess.run(valid_iterator.initializer, feed_dict={validData_placeholder: dataset['valid']})
            while True:
                try:
                    x = self._sess.run(next_valid_batch)
                    validLoss.append(x.shape[0] * self.val_function(x))
                except tf.errors.OutOfRangeError:
                    break
            validLoss_avg = np.asarray(validLoss).sum() / len(dataset['valid'])

            testLoss = []
            self._sess.run(test_iterator.initializer, feed_dict={testData_placeholder: dataset['test']})
            while True:
                try:
                    x = self._sess.run(next_test_batch)
                    testLoss.append(x.shape[0] * self.val_function(x))
                except tf.errors.OutOfRangeError:
                    break
            testLoss_avg = np.asarray(testLoss).sum() / len(dataset['test'])

            # evaluate the model w.r.t the valid set and record the average loss.
            print("BEST MODEL from epoch \x1b[1;91m%4d\x1b[0m with training loss"
                  " \x1b[1;91m%10.4f\x1b[0m and valid loss \x1b[1;91m%10.4f\x1b[0m."
                  % (bestEpoch, trainLoss_avg, validLoss_avg))
            print('The testing loss is \x1b[1;91m%10.4f\x1b[0m.' % testLoss_avg)
            # save the record as npz.
            np.savez(saveto, historyLoss=np.asarray(historyLoss),
                     testLoss=testLoss_avg, durationEpoch=durationEpoch)
        return
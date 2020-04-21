from __future__ import absolute_import,division,print_function

import gzip
import os
import time
import sys
import math
from six.moves import xrange
import numpy as np
from CNN import CNN
from data_helper import load_train,load_test

# hyper parameter
IMAGE_SIZE=28
NUM_CHANNELS=1 
PIXEL_DEPTH=255
BATCH_SIZE=64
NUM_LABELS=10
VAL_SIZE=64
MAX_EPOCHS = 2
SEED=2330
EVAL_FREQUENCY = 50
EVAL_BATCH_SIZE = 64

def onehot(targets, num):
    result = np.zeros((num, 10))
    for i in range(num):
        result[i][targets[i]] = 1
    return result

validation_size = 5000
train_data,train_labels,validation_data,validation_labels = load_train(validation_size,IMAGE_SIZE, NUM_CHANNELS, PIXEL_DEPTH)
test_data,test_labels = load_test(IMAGE_SIZE, NUM_CHANNELS, PIXEL_DEPTH)

train_labels = onehot(train_labels, 55000)
validation_labels = onehot(validation_labels, validation_size)
train_size = train_data.shape[0]

# calculate accuracy
def cal_acc(predictions, labels):
    return (100.0 * np.sum(predictions == labels) /predictions.shape[0]) 

# predict validation data and test data
def eval_in_batches(data,cnn):
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
        raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = np.zeros(size)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
        end = begin + EVAL_BATCH_SIZE
        if end <= size:
            x = data[begin:end, ...]
            predictions[begin:end] = cnn.cal_forward(x)
        else:
            x = data[-EVAL_BATCH_SIZE:, ...]
            batch_predictions = cnn.cal_forward(x)
            predictions[begin:] = batch_predictions[begin - size:]
    return predictions


if __name__ == "__main__":
    cnn = CNN(num_labels=10, batch_size=64, image_size=28, num_channels=1, seed=SEED)
    learning_rate = 0.01

    start_time = time.time()
    # train 
    print("="*20, "start training","="*20)
    for step in xrange(int(MAX_EPOCHS * train_size) // BATCH_SIZE):
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_x = train_data[offset:(offset + BATCH_SIZE), ...]
        batch_y = train_labels[offset:(offset + BATCH_SIZE)]

        loss, predictions = cnn.fit(batch_x,batch_y,learning_rate)
            
        if step % EVAL_FREQUENCY == 0:
            acc = cal_acc(predictions, np.argmax(batch_y, axis=1))
            print('Step %d (epoch %.2f)' % (step, float(step) * BATCH_SIZE / train_size))
            print('Minibatch loss: %.3f' % loss)
            print('Minibatch acc:  %.1f%%' % acc)
            val_predictions = eval_in_batches(validation_data[0:1000, ...],cnn)
            val_acc = cal_acc(val_predictions, np.argmax(validation_labels[0:1000], axis=1))
            print('Validation error: %.1f%%' % val_acc)

    # test
    print("="*20, "start testing","="*20)
    test_predictions = eval_in_batches(test_data, cnn)
    test_acc = cal_acc(test_predictions, test_labels)
    print("test data acc: %.1f%%" % test_acc)
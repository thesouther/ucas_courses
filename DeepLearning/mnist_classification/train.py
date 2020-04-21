from __future__ import absolute_import,division,print_function
import argparse
import gzip
import os
import sys
import time
import argparse

import numpy as np
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf
from data_helper import download_data,extract_data,extract_labels
import CNN

# use CPU default
os.environ["CUDA_VISIBLE_DEVICES"]="1"

parser = argparse.ArgumentParser(description='CNN by CC')
parser.add_argument('-e','--epochs', type=int, default=10,
                    help='number of epochs for train')
parser.add_argument('-b','--batch-size', type=int, default=64,
                    help='batch size for training')
parser.add_argument('-f','--eval-frequency', type=int, default=100,
                    help='evaluation frequency')
parser.add_argument('-o','--optimizer', type=str, default='momentum',choices=['momentum','adam'],
                    help='optimizer, you can select from [momentum, adam]')
parser.add_argument('-s','--save', action='store_true',
                    help='path to save the final model')
parser.add_argument('-g','--use-gpu', action='store_true',
                    help='select do or don\'t use GPU for training')
parser.add_argument('-v','--validation-size', type=int, default=5000,
                    help='validation data size')

args = parser.parse_args()

SAVE = False
SAVE_DIR = 'model'
DOWNLOAD=False

WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
EVAL_BATCH_SIZE = 64
SEED = 66478
NUM_LABELS = 10

if args.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
if args.save:
    SAVE = True
    SVAV_DIR = 'model/cnn_model.ckpt'

def load_data():
    file_directory = WORK_DIRECTORY
    validation_size = args.validation_size

    # get data
    train_data_filename = download_data(file_directory, "train-images-idx3-ubyte.gz", download=DOWNLOAD)
    train_labels_filename = download_data(file_directory, "train-labels-idx1-ubyte.gz", download=DOWNLOAD)

    train_data = extract_data(train_data_filename, 60000, IMAGE_SIZE, NUM_CHANNELS, PIXEL_DEPTH)
    train_labels = extract_labels(train_labels_filename,60000)

    validation_data = train_data[:validation_size, ...]
    validation_labels = train_labels[:validation_size]
    train_data = train_data[validation_size:, ...]
    train_labels = train_labels[validation_size:]
    return train_data,train_labels,validation_data,validation_labels

Return the error rate based on dense predictions and sparse labels.
def error_rate(predictions, labels):
  return 100.0 - (
      100.0 *
      np.sum(np.argmax(predictions, 1) == labels) /
      predictions.shape[0])

def main(_):
    train_data, train_labels, validation_data, validation_labels = load_data()
    # print(len(train_data))
    num_epochs = args.epochs
    batch_size = args.batch_size
    eval_frequency = args.eval_frequency
    num_train = len(train_data)
    epoch_steps = num_train // batch_size

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    with tf.name_scope("input"):
        X = tf.placeholder(tf.float32,shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        y = tf.placeholder(tf.int64, shape=(batch_size,))
    with tf.name_scope('eval_input'):
        eval_X = tf.placeholder(tf.float32, shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    with tf.name_scope('cnn_model'):
        cnn = CNN.CNN(NUM_LABELS,batch_size,EVAL_BATCH_SIZE,IMAGE_SIZE,NUM_CHANNELS,SEED)
 
    logits,regularizers = cnn.model(X, drop_out=True)
    with tf.name_scope('loss'):  
    # L2 + cross-entropy loss
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)) + (5e-4*regularizers)
        tf.summary.scalar('loss',loss)
    
    batch = tf.Variable(0, dtype=tf.float32)
    lr = tf.train.exponential_decay(
        0.01,                # Base learning rate.
        batch * batch_size,  # Current index into the dataset.
        num_train,           # Decay step.
        0.95,                # Decay rate.
        staircase=True)
    with tf.name_scope('train'):
        if args.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss, global_step=batch)
        else:
            optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
    
    train_prediction = tf.nn.softmax(logits)
    eval_prediction = tf.nn.softmax(cnn.model(eval_X, drop_out=False))

    # Get all predictions for a dataset by running it in small batches.
    def eval_in_batches(data, sess):
        size = data.shape[0]
        if size < EVAL_BATCH_SIZE:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)
        for begin in xrange(0, size, EVAL_BATCH_SIZE):
            end = begin + EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run(eval_prediction,feed_dict={eval_X: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(eval_prediction,feed_dict={eval_X: data[-EVAL_BATCH_SIZE:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions
 
    # start training 
    start_time = time.time()
    with tf.Session() as sess:
        # write loss into logs
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs/', sess.graph)
        tf.global_variables_initializer().run()
        print("Initialized! Start Training")
        for epc in xrange(num_epochs):
            print("="*20,"epoch: ", str(epc),"="*20)
            for step in xrange(epoch_steps):
                offset = step * batch_size
                batch_data = train_data[offset : (offset+batch_size), ...]
                batch_labels = train_labels[offset : (offset+batch_size)]
                feed_dict = {X:batch_data,y:batch_labels}
                sess.run(optimizer, feed_dict=feed_dict)
                # print centre results.
                if (step) % eval_frequency == 0:
                    l,lar,predictions = sess.run([loss, lr, train_prediction], feed_dict=feed_dict)
                    
                    result = sess.run(merged,feed_dict=feed_dict)
                    writer.add_summary(result, epc*epoch_steps + step)

                    run_time = time.time() - start_time
                    print("step: %d, run_time: %.1f ms" % (step, run_time))
                    print('  Minibatch loss: %.3f, learning rate: %.6f' % (l, lar))
                    print('  Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
                    print('  Validation error: %.1f%%' % error_rate(eval_in_batches(validation_data, sess), validation_labels))
                    sys.stdout.flush()
        if SAVE:
            #Create a saver object which will save all the variables
            saver = tf.train.Saver()
            saver.save(sess, SAVE_DIR+'/cnn_model.ckpt')

if __name__ == "__main__":
    tf.app.run()
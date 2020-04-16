from __future__ import absolute_import,division,print_function
import os
import sys

import numpy as np
from six.moves import xrange
import tensorflow as tf
from data_helper import download_data,extract_data,extract_labels
import CNN

# use CPU default
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

MODEL_PATH = './model'
WORK_DIRECTORY = 'data'
DOWNLOAD=False

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
EVAL_BATCH_SIZE = 64
SEED = 66478
NUM_LABELS = 10

tf.reset_default_graph()

# compute error rate
def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      np.sum(np.argmax(predictions, 1) == labels) /
      predictions.shape[0])

def load_data():
    file_directory = WORK_DIRECTORY
    # get data
    test_data_filename = download_data(file_directory, "t10k-images-idx3-ubyte.gz", download=DOWNLOAD)
    test_labels_filename = download_data(file_directory, "t10k-labels-idx1-ubyte.gz", download=DOWNLOAD)
    test_data = extract_data(test_data_filename, 10000, IMAGE_SIZE, NUM_CHANNELS, PIXEL_DEPTH)
    test_labels = extract_labels(test_labels_filename, 10000)
    return test_data,test_labels

def main(_):
    test_data,test_labels = load_data()
    test_size = len(test_data)
    # print(test_size)

    test_X = tf.placeholder(tf.float32, shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

    with tf.name_scope('cnn_model'):
        cnn = CNN.CNN(NUM_LABELS,EVAL_BATCH_SIZE,EVAL_BATCH_SIZE,IMAGE_SIZE,NUM_CHANNELS,SEED)
    eval_prediction = tf.nn.softmax(cnn.model(test_X, drop_out=False))

    # Get all predictions for a dataset by running it in small batches.
    def eval_in_batches(data, sess):
        size = data.shape[0]
        if size < EVAL_BATCH_SIZE:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)
        for begin in xrange(0, size, EVAL_BATCH_SIZE):
            end = begin + EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run(eval_prediction,feed_dict={test_X: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(eval_prediction,feed_dict={test_X: data[-EVAL_BATCH_SIZE:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions
    
    # test process
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # reload model
        ckpt = tf.train.latest_checkpoint(MODEL_PATH)
        saver.restore(sess, ckpt)

        predictions = eval_in_batches(test_data,sess)
        test_error = error_rate(predictions, test_labels)
        acc = 100.0 - test_error

        print('accuracy: %.1f%%' % acc)

if __name__ == "__main__":
    tf.app.run()
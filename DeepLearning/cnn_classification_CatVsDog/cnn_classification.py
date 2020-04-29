import os
import sys
import time
from six.moves import xrange
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from data_helper import extract_data
from CNN import CNN

os.environ["CUDA_VISIBLE_DEVICES"]="2"

IMAGE_SIZE=128
NUM_CHANNELS=3
NUM_LABELS=2

NUM_EPOCHS=50
EVAL_FREQUENCY=100
BATCH_SIZE=64
OPTIMIZER='adam'
VALIDATION_SIZE = 2000
SEED = 66478
classes = ['dog', 'cat']

def cal_accuracy(predictions, labels):
  return 100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0]

def plot_curve(x_len,y1,y2,img_name):
    steps = np.arange(x_len) 
    plt.figure()
    plt.plot(steps, y1, label='train accuracy')
    plt.plot(steps, y2, label='validition accuracy')
    plt.title("accuracy curve")
    plt.xlabel('steps / 100')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')
    plt.savefig(img_name)


def main(_):
    # extract data and divided into train_data and val_data
    data_image,data_label =extract_data(IMAGE_SIZE, data_type='train')
    train_data = data_image[VALIDATION_SIZE:]
    train_label = data_label[VALIDATION_SIZE:]
    val_data = data_image[:VALIDATION_SIZE]
    val_label = data_label[:VALIDATION_SIZE]
    num_train = len(train_data)
    # extract test data and test lables
    test_data,test_label = extract_data(IMAGE_SIZE, data_type='test')
    
    num_epochs = NUM_EPOCHS
    batch_size = BATCH_SIZE
    eval_frequency = EVAL_FREQUENCY
    optimizer =OPTIMIZER
    
    with tf.name_scope("input"):
        X = tf.placeholder(tf.float32,shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        y = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
    with tf.name_scope('eval_input'):
        eval_X = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        
    # Instantiation CNN model
    with tf.name_scope('cnn_model'):
        cnn = CNN(batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, num_channals=NUM_CHANNELS, num_labels=NUM_LABELS, seed=SEED)
 
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
        if optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss, global_step=batch)
        else:
            optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
    
    train_prediction = tf.nn.softmax(logits)
    eval_prediction = tf.nn.softmax(cnn.model(eval_X, drop_out=False))

    def eval_in_batches(data, sess):
        """Get all predictions for a dataset by running it in small batches."""
        size = data.shape[0]
        if size < BATCH_SIZE:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)
        for begin in xrange(0, size, BATCH_SIZE):
            end = begin + BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run(eval_prediction,feed_dict={eval_X: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(eval_prediction,feed_dict={eval_X: data[-BATCH_SIZE:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions
 
    with tf.Session() as sess:
        # Visualize graph
        # write loss into logs
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./logs/', sess.graph)
        tf.global_variables_initializer().run()
        val_accs = []
        train_accs = []
        
        print("Initialized! Start Training")
        for step in xrange(int(num_epochs * num_train) // batch_size):
            offset = (step * batch_size) % (num_train - batch_size)
            batch_data = train_data[offset:(offset + batch_size), ...]
            batch_label = train_label[offset:(offset + batch_size)]
            feed_dict = {X:batch_data,y:batch_label}
            sess.run(optimizer, feed_dict=feed_dict)
            if (step) % eval_frequency == 0:
                # fetch some extra nodes' data
                l, predictions = sess.run([loss, train_prediction],feed_dict=feed_dict)
                result = sess.run(merged,feed_dict=feed_dict)
                writer.add_summary(result, step)

                train_acc = cal_accuracy(predictions, batch_label)
                val_acc = cal_accuracy(eval_in_batches(val_data, sess), val_label)
                train_accs.append(train_acc)
                val_accs.append(val_acc)

                print('Step %d (epoch %.2f)' % (step, float(step) * batch_size / num_train))
                print('Minibatch loss: %.3f' % (l))
                print('Minibatch accuracy: %.1f%%' % train_acc)
                print('Validation accuracy: %.1f%%' % val_acc)
                sys.stdout.flush()

        # plot accuracy curve
        acc_save_name = "./result/acc.png"
        plot_curve(len(train_accs),train_accs,val_accs,acc_save_name)

        # calculate test accuracy
        test_acc  = cal_accuracy(eval_in_batches(test_data, sess), test_label)
        print('Test accuracy: %.1f%%' % test_acc)

if __name__ == "__main__":
    tf.app.run()
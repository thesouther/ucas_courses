import os
import time
import sys
import tensorflow as tf
import numpy as np

from data_helper import load_data,pick_word_index
from model import RNN

os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

SAVE=True
CUR_DIR = os.getcwd()
CKPT_PATH = CUR_DIR + '/model/'

# hyper parameter
NUM_EPOCH = 10
BATCH_SIZE = 64
len_vector = 125
embedding_dim = 256
n_neurons = embedding_dim
n_layers = 3
lr = 0.001
keep_prob=0.8
eval_frequence = 100
model='lstm' # lstm, gru, rnn

def generate_poem(rnn_model, sess, input_seqs, ix2word, word2ix, max_len=125, prefix_words=None):
    len_sqes = len(input_seqs)
    result = list(input_seqs)
    # build input word vector
    input_word = np.zeros((1,1), dtype=np.int32)
    input_word[0,0] = word2ix['<START>']
    # placeholders for sess to run
    test_X=tf.placeholder(tf.int32, [1, None])
    prediction, output_state = rnn_model.generate_word(test_X)
    last_state = sess.run(rnn_model.initial_state)
    # if prefix style of poem is not None, pretrain the model use prefix_word
    if prefix_words:
        for word in prefix_words:
            prob, last_state = sess.run([prediction, output_state], feed_dict={test_X:input_word, rnn_model.initial_state:last_state})
            input_word[0,0] = word2ix[word]
    # generate poem
    for i in range(max_len):
        prob, last_state = sess.run([prediction, output_state], feed_dict={test_X:input_word, rnn_model.initial_state:last_state})
        if i < len_sqes:
            word = result[i]
        else:
            word_idx = pick_word_index(prob)
            word = ix2word[word_idx]
            result.append(word)
        input_word[0,0] = word2ix[word]
        if word=='<EOP>':
            del result[-1]
            break
    return result

def main(_):
    # load data
    data, ix2word, word2ix = load_data()
    num_train = data.shape[0]
    vocab_size = len(ix2word)
    # variables for training
    X=tf.placeholder(tf.int32, [BATCH_SIZE, None])
    y=tf.placeholder(tf.int32, [BATCH_SIZE, None])
    rnn_model = RNN(model=model, batch_size=BATCH_SIZE, vocab_size=vocab_size, embedding_dim=embedding_dim, n_neurons=n_neurons, n_layers=3, lr=lr, keep_prob=keep_prob)
    loss, optimizer = rnn_model.train(X, y)

    # start trian
    start_time = time.time()
    with tf.Session() as sess:
        # Visualize graph
        # write loss into logs
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./logs/', sess.graph)
        tf.global_variables_initializer().run()
        print("="*15+"strat training"+"="*15)
        for epc in range(NUM_EPOCH):
            print("="*15, "epoch: %d" % epc, "="*15)
            for step in range(num_train//BATCH_SIZE):
                # get batch data
                idx_strat = step*BATCH_SIZE
                idx_end = idx_strat+BATCH_SIZE
                batch_data = data[idx_strat:idx_end, ...]
                x_data = batch_data[:, :-1]
                y_data = batch_data[:, 1:]

                feed_dict={X:x_data,y:y_data}
                sess.run(optimizer, feed_dict=feed_dict)
                
                # print evaluation results for every 100 steps
                if step%eval_frequence==0:
                    l = sess.run(loss,feed_dict=feed_dict)
                    result = sess.run(merged,feed_dict=feed_dict)
                    writer.add_summary(result, (epc*num_train//BATCH_SIZE)+step)

                    input_seq = "湖光秋月两相和"
                    result = generate_poem(rnn_model=rnn_model, sess=sess, input_seqs=input_seq, ix2word=ix2word,word2ix=word2ix, max_len=125, prefix_words=None)
                    result_poem = ''.join(result)
                    
                    run_time = time.time() - start_time
                    start_time = time.time()
                    print("step: %d, run time: %.1f ms" % (step, run_time*1000/eval_frequence))
                    print("minibatch loss: %d" % l)
                    print("generated poem length: %d, poem is: %s" % (len(result_poem), result_poem))
                    sys.stdout.flush()
        # save model
        if SAVE:
            saver = tf.train.Saver()
            saver.save(sess, CKPT_PATH+'rnn_model.ckpt')

if __name__ == "__main__":
    tf.app.run()

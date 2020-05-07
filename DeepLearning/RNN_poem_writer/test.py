import numpy as np
import tensorflow as tf
import os
from data_helper import load_data,pick_word_index
from model import RNN
# from train import generate_poem

os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
CUR_DIR = os.getcwd()
CKPT_PATH = CUR_DIR + '/model'

# change input_seq to generate different poems
input_seq = "湖光秋月两相和"
BATCH_SIZE = 1
max_len=(len(input_seq)+1)*4
prefix_words=''

len_vector = 125
embedding_dim = 256
n_neurons = embedding_dim
n_layers = 3
lr = 0.01
keep_prob=0.8
eval_frequence = 100
model='lstm' # lstm, gru, rnn

def main(_):
    len_sqes = len(input_seq)
    result = list(input_seq)

    _, ix2word, word2ix = load_data()
    vocab_size = len(ix2word)
    
    rnn_model = RNN(model=model, batch_size=BATCH_SIZE, vocab_size=vocab_size, embedding_dim=embedding_dim, n_neurons=n_neurons, n_layers=3, lr=lr, keep_prob=keep_prob)
    
    # build input word vector
    input_word = np.zeros((1,1), dtype=np.int32)
    input_word[0,0] = word2ix['<START>']
    # placeholders for sess to run
    test_X=tf.placeholder(tf.int32, [1, None])
    prediction, output_state = rnn_model.generate_word(test_X)

    saver = tf.train.Saver()
    # generate poem
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print("="*15+"generating poem"+"="*15)
        ckpt = tf.train.latest_checkpoint(CKPT_PATH)
        saver.restore(sess, ckpt)

        last_state = sess.run(rnn_model.initial_state)
        # if prefix style of poem is not None, pretrain the model use prefix_word
        if len(prefix_words)>0:
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

        result_poem = ''.join(result)
        print("generated poem length: %d, poem is: %s" % (len(result_poem), result_poem))

if __name__ == "__main__":
    tf.app.run()

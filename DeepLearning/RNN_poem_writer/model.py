import tensorflow as tf
import numpy as np
import pdb  
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class RNN:
    def __init__(self, model='lstm', batch_size=64, vocab_size=8293, embedding_dim=256, n_neurons=256, n_layers=2, lr=0.01, keep_prob=0.5):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.lr = lr
        self.keep_prob = keep_prob

        self.embedding = tf.get_variable('embedding', initializer=tf.random_uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0))

        if model=='rnn': # LSTM, RNN, GRU
            cell_rnn = tf.contrib.rnn.BasicRNNCell
        elif model == 'lstm':
            cell_rnn = tf.contrib.rnn.BasicLSTMCell
        elif model == 'gru':
            cell_rnn = tf.contrib.rnn.GRUCell
        cell = cell_rnn(num_units=n_neurons)
        drop = tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=self.keep_prob)
        self.multi_cell = tf.contrib.rnn.MultiRNNCell([drop] * self.n_layers)
        # weights of full connect layer
        self.fc_w = tf.Variable(tf.truncated_normal([self.n_neurons, self.vocab_size]))
        self.fc_b = tf.Variable(tf.zeros(shape=[self.vocab_size]))
            
    def train(self, X, y):
        # ==========embedding input===========
        with tf.name_scope("input"):
            inputs = tf.nn.embedding_lookup(self.embedding, X) # (64, 124, 256)
        # pdb.set_trace()  

        # ===============RNN layers===========
        with tf.name_scope('RNN'):
            initial_state = self.multi_cell.zero_state(self.batch_size, tf.float32)
            outputs, states = tf.nn.dynamic_rnn(self.multi_cell, inputs, initial_state=initial_state)
        
        # =============fc layer================
        with tf.name_scope('FC'):
            output = tf.reshape(outputs, [-1, self.n_neurons])
            logits = tf.nn.bias_add(tf.matmul(output, self.fc_w), bias=self.fc_b)

        # ========calculate loss function======
        with tf.name_scope('loss'):
            labels = tf.one_hot(tf.reshape(y, [-1]), depth=self.vocab_size) # shape=[?, vocab_size]
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
            tf.summary.scalar('loss',loss)
        # ==============Optimizer==============
        with tf.name_scope('AdamOpt'):
            optimizer = tf.train.AdamOptimizer(self.lr).minimize(loss)

        return loss, optimizer

    def generate_word(self, X):
        # embedding input word
        input_word = tf.nn.embedding_lookup(self.embedding, X) # (1,1,256)
        # get hidden state
        self.initial_state = self.multi_cell.zero_state(1, tf.float32)
        output_word, output_state = tf.nn.dynamic_rnn(self.multi_cell, input_word, initial_state=self.initial_state)
        output = tf.reshape(output_word, [-1, self.n_neurons])
        logits = tf.nn.bias_add(tf.matmul(output, self.fc_w), bias=self.fc_b)
        prediction = tf.nn.softmax(logits)
        return prediction, output_state

# test
if __name__ == "__main__":
    from data_helper import load_data,pick_word_index
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="2"

    def generate_poem(model, sess, input_seqs, ix2word, word2ix, max_len=125, prefix_words=None):
        len_sqes = len(input_seqs)
        result = list(input_seqs)
        # build input word vector
        input_word = np.zeros((1,1), dtype=np.int32)
        input_word[0,0] = word2ix['<START>']

        test_X=tf.placeholder(tf.int32, [1, None])
        prediction, output_state = model.generate_word(test_X)
        last_state = sess.run(model.initial_state)

        # if prefix style of poem is not None, pretrain the model use prefix_word
        if prefix_words:
            for word in prefix_words:
                prob, last_state = sess.run([prediction, output_state], feed_dict={test_X:input_word, model.initial_state:last_state})
                input_word[0,0] = word2ix[word]

        # generate poem
        for i in range(max_len):
            prob, last_state = sess.run([prediction, output_state], feed_dict={test_X:input_word, model.initial_state:last_state})
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

    data, ix2word, word2ix = load_data()
    num_train = data.shape[0]
    vocab_size = len(ix2word)
    batch_size=64

    idx_strat = 0* 64
    idx_end = idx_strat+64
    batch_data = data[idx_strat:idx_end, ...]
    x_data = batch_data[:, :-1]
    y_data = batch_data[:, 1:]

    X=tf.placeholder(tf.int32, [batch_size, None])
    y=tf.placeholder(tf.int32, [batch_size, None])

    rnn_model = RNN(model='lstm', batch_size=batch_size, vocab_size=vocab_size, embedding_dim=256, n_neurons=256, n_layers=3, lr=0.01)
    loss, optimizer = rnn_model.train(X, y)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        feed_dict={X:x_data,y:y_data}
        sess.run(optimizer, feed_dict=feed_dict)
        l = sess.run(loss,feed_dict=feed_dict)
        print("loss %f " % l)
        input_seq = "湖光秋月两相和"
        result = generate_poem(rnn_model, sess, input_seq, ix2word, word2ix, max_len=125, prefix_words=None)
        print(result)
        



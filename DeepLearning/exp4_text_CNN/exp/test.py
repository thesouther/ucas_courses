# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
from config import Config
conf = Config()
from data_helper import build_word2id,build_word2vec,load_corpus,cal_accuracy,eval_in_batches

import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

if conf.select_model == 'model2':    
    from model2 import TextCNN 
else:
    from model import TextCNN 

def main(_):
    word2id = build_word2id(conf.word2id_path)
    test_data,test_labels = load_corpus(conf.test_path, word2id, max_sen_len=conf.max_sen_len)
    print("test data shape: ", test_data.shape)
    word_vecs= build_word2vec(conf.pre_word2vec_path, word2id, conf.corpus_word2vec_path)
    test_X = tf.placeholder(tf.int64, shape=(conf.batch_size, conf.max_sen_len))
    with tf.name_scope("TextCNN"):
        text_cnn = TextCNN(word_vecs)
    test_prediction = tf.nn.softmax(text_cnn.model(test_X, train=False))

    # test process
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # reload model
        
        ckpt = tf.train.latest_checkpoint(conf.save_dir)
        saver.restore(sess, ckpt)

        predictions = eval_in_batches(test_data,test_prediction, test_X, sess) # data, eval_prediction, eval_data, sess
        test_acc = cal_accuracy(predictions, test_labels)

        print('test accuracy: %.1f%%' % test_acc)
          
if __name__ == "__main__":
    tf.app.run()

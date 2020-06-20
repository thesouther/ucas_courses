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
    train_data,train_labels = load_corpus(conf.train_path, word2id, max_sen_len=conf.max_sen_len)
    val_data, val_labels = load_corpus(conf.val_path, word2id, max_sen_len=conf.max_sen_len)
    word_vecs= build_word2vec(conf.pre_word2vec_path, word2id, conf.corpus_word2vec_path)
    # print("val ", val_data.shape)
    # print(val_data[10])
    
    train_size = train_data.shape[0]
    print("train_size: ",train_size)

    with tf.name_scope("input"):
        X=tf.placeholder(tf.int64, shape=(conf.batch_size, conf.max_sen_len))
        y=tf.placeholder(tf.int64, shape=(conf.batch_size,))
    eval_X = tf.placeholder(tf.int64, shape=(conf.batch_size, conf.max_sen_len))
    
    with tf.name_scope("TextCNN"):
        text_cnn = TextCNN(word_vecs)

    output = text_cnn.model(X, train=True)

    with tf.name_scope("loss"):
        # 这里的logits通常是最后的全连接层的输出结果
        # labels是具体哪一类的标签，这个函数是直接使用标签数据的，而不是采用one-hot编码形式。
        loss =tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=output))
        tf.summary.scalar('loss',loss)

    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(conf.learning_rate).minimize(loss)

    train_prediction = tf.nn.softmax(output)
    eval_prediction = tf.nn.softmax(text_cnn.model(eval_X, train=False))

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs/', sess.graph)
        tf.global_variables_initializer().run()
        print('Initialized!')

        for step in range(int(conf.n_epoch * train_size) // conf.batch_size):
            offset = (step * conf.batch_size) % (train_size-conf.batch_size)
            
            batch_data = train_data[offset: (offset+conf.batch_size)] # (100,75)
            batch_labels = train_labels[offset: (offset+conf.batch_size)]
            
            feed_dict = {X:batch_data,y:batch_labels}
            sess.run(optimizer, feed_dict=feed_dict)
            if step % conf.print_per_batch ==0:
                l, pred = sess.run([loss, train_prediction], feed_dict=feed_dict)
                result = sess.run(merged,feed_dict=feed_dict)
                writer.add_summary(result, step)
                print("="*20, "step: %d" % step, "="*20)
                print("minibatch loss: %.3f " % l)
                print("minibatch acc: %.1f%%" % cal_accuracy(pred, batch_labels))
                val_pred = eval_in_batches(val_data, eval_prediction, eval_X, sess)
                print("validation acc: %.1f%%" % cal_accuracy(val_pred, val_labels))
        if conf.save:
            saver = tf.train.Saver()
            print(conf.save_dir+"text_cnn.ckpt")
            saver.save(sess, conf.save_dir+"text_cnn.ckpt")


if __name__ == "__main__":
    tf.app.run()
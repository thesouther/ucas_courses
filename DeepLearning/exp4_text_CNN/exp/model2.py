# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np

from config import Config
conf = Config()

class TextCNN:
    def __init__(self,word_vecs):
        self.total_filters_len = conf.kernel_classes*conf.num_filters
        self.word_vecs = word_vecs
        self.update_w2v = conf.update_w2v
        self.w_conv = []
        self.b_conv = []
        for i, filter_size in enumerate(conf.kernel_size): # kernel size = [2,3,4,5]
            w = self.weight_variable([filter_size, conf.embedding_dim, conf.input_channel, conf.num_filters],conf.seed)
            b = self.bias_variable([conf.num_filters])
            self.w_conv.append(w)
            self.b_conv.append(b)
        # self.pool = []
        self.w_fc1 = self.weight_variable([self.total_filters_len, conf.n_class], conf.seed)
        # self.w_fc1 = self.weight_variable([conf.num_filters, conf.n_class], conf.seed)
        self.b_fc1 = self.bias_variable([conf.n_class])
        self.embedding = tf.get_variable('embedding', shape=[conf.vocab_size, conf.embedding_dim], initializer=tf.constant_initializer(self.word_vecs))#, trainable=self.update_w2v)

    def weight_variable(self, shape, seed):
        initial = tf.truncated_normal(shape, stddev=0.1, seed=seed, dtype=tf.float32)
        return tf.Variable(initial)
    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
        return tf.Variable(initial)

    def model(self,input_x, train=True):
        # print("input_x.shape",input_x.shape) # input_x.shape (100, 75)
        with tf.name_scope('input'):
            X = tf.nn.embedding_lookup(self.embedding, input_x)
        # print("X.shape",X.shape) # X.shape (100, 75, 50)
        X = tf.expand_dims(X,-1)
        # X = tf.reshape(X,[conf.batch_size, conf.max_sen_len, conf.embedding_dim, conf.input_channel])
        # print("X.shape",X.shape) # X.shape (100, 75, 50, 1)
        
        pool_out=[]
        for i, filter_size in enumerate(conf.kernel_size): # kernel size = [2,3,4,5]
            with tf.name_scope('conv-with-kernel-%s' % filter_size):
                # W = tf.Variable(tf.truncated_normal([filter_size, conf.embedding_dim, conf.input_channel, conf.num_filters], stddev=0.1), name="W")
                # b = tf.Variable(tf.constant(0.1, shape=[conf.num_filters]), name="b")
                conv1 = tf.nn.conv2d(X, self.w_conv[i], strides=[1,1,1,1], padding='VALID')
                # print("conv1: ", conv1.shape) # conv1:  (100, 73, 1, 256)
                relu1 = tf.nn.relu(tf.nn.bias_add(conv1, self.b_conv[i]))
                # print("relu1",relu1.shape) # relu1 (100, 73, 1, 256)
                pool1 = tf.nn.max_pool(relu1, ksize=[1,conf.max_sen_len-filter_size+1,1,1], strides=[1,1,1,1], padding='VALID')
                # print("pool1", pool1.shape) # pool1 (100, 1, 1, 256)
                pool_out.append(pool1)
        
        with tf.name_scope("flatten"):
            self.total_pool = tf.concat(pool_out, 3)
            flatten = tf.reshape(self.total_pool, [-1,self.total_filters_len])
            # print("pool_reshape", flatten.shape) # pool_reshape (100, 256)

        with tf.name_scope("full_connected"):
            fc1 = tf.matmul(flatten,self.w_fc1) + self.b_fc1
            # print("fc1", fc1.shape) # fc1 (100, 2)

        return fc1

from __future__ import absolute_import,division,print_function

import tensorflow as tf

class CNN:
    def __init__(self,num_labels=10, batch_size=64, eval_batch_size=64, image_size=28, num_channels=1, seed=66478):
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.seed = seed
        self.num_labels=num_labels
        self.eval_batch_size = eval_batch_size

        # paramater definition
        self.w_conv1 = self.weight_variable([5,5,self.num_channels,32],self.seed) # 5*5 filter
        self.b_conv1 = self.bias_variable([32])
        self.w_conv2 = self.weight_variable([5,5,32,64],self.seed)
        self.b_conv2 = self.bias_variable([64])
        self.w_fc1 = self.weight_variable([self.image_size//4 * self.image_size//4 * 64, 512], self.seed)
        self.b_fc1 = self.bias_variable([512])
        self.w_fc2 = self.weight_variable([512, self.num_labels], self.seed)
        self.b_fc2 = self.bias_variable([self.num_labels])

    # weight definition.
    def weight_variable(self, shape, seed):
        initial = tf.truncated_normal(shape, stddev=0.1, seed=seed, dtype=tf.float32)
        return tf.Variable(initial)

    # bias difinition
    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
        return tf.Variable(initial)
    

    # buil CNN model
    def model(self, X, drop_out=False):
        seed = self.seed
        # image_size = self.image_size
        # num_labels = self.num_labels

        # ********************** conv1 *********************************
        # transfer a 5*5*1 imagine into 32 sequence  
        with tf.name_scope('conv1'):
            with tf.name_scope('weight'):
                w_conv1 = self.w_conv1
            with tf.name_scope('bias'):
                b_conv1 = self.b_conv1
            with tf.name_scope('conv_and_pool'):
                conv1 = tf.nn.conv2d(X, w_conv1, strides=[1,1,1,1],padding='SAME')
                relu1 = tf.nn.relu(tf.nn.bias_add(conv1, b_conv1))
                pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        # ********************** conv2 *********************************
        # transfer a 5*5*1 imagine into 32 sequence  
        with tf.name_scope('conv2'):
            with tf.name_scope('weight'):
                w_conv2 = self.w_conv2
            with tf.name_scope('bias'):
                b_conv2 = self.b_conv2
            with tf.name_scope('conv_and_pool'):
                conv2 = tf.nn.conv2d(pool1, w_conv2, strides=[1,1,1,1],padding='SAME')
                relu2 = tf.nn.relu(tf.nn.bias_add(conv2, b_conv2))
                pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        # ********************* func1 layer *********************************
        # reshape the image from 7,7,64 into a flat (7*7*64),
        # and compute result
        with tf.name_scope('fc1'):
            with tf.name_scope('weight'):
                w_fc1 = self.w_fc1
            with tf.name_scope('bias'):
                b_fc1 = self.b_fc1
            with tf.name_scope('flatting'):    
                pool_shape = pool2.get_shape().as_list()
                reshape = tf.reshape(pool2, [pool_shape[0], pool_shape[1]*pool_shape[2]*pool_shape[3]])
                fc1 = tf.nn.relu(tf.matmul(reshape, w_fc1) + b_fc1)

        # ********************* func2 layer *********************************
        # if it is training, use drop out
        # else, if validating, don't use drop out!
        with tf.name_scope('fc2'):
            with tf.name_scope('weight'):
                w_fc2 = self.w_fc2
            with tf.name_scope('bias'):
                b_fc2 = self.b_fc2
            if drop_out:
                fc1 = tf.nn.dropout(fc1, 0.5, seed=seed)
                fc2 = tf.matmul(fc1, w_fc2) + b_fc2
                regularizers = (tf.nn.l2_loss(w_fc1) + tf.nn.l2_loss(b_fc1) + 
                            tf.nn.l2_loss(w_fc2) + tf.nn.l2_loss(b_fc2))
                return fc2,regularizers
            else:
                return tf.matmul(fc1, w_fc2) + b_fc2


        
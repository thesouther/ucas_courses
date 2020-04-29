from __future__ import absolute_import,division,print_function
import tensorflow as tf

class CNN:
    def __init__(self, batch_size=64, image_size=64, num_channals=3, num_labels=2, seed=66478):
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channals = num_channals
        self.num_labels = num_labels
        self.seed = seed
        
        self.w_conv1 = self.buil_weight([3,3,self.num_channals,32], self.seed)
        self.b_conv1 = self.buil_bias([32])
        self.w_conv2 = self.buil_weight([3,3,32,32], self.seed)
        self.b_conv2 = self.buil_bias([32])
        self.w_conv3 = self.buil_weight([3,3,32,64], self.seed)
        self.b_conv3 = self.buil_bias([64])
        self.w_conv4 = self.buil_weight([3,3,64,64], self.seed)
        self.b_conv4 = self.buil_bias([64])
        self.w_fc1 = self.buil_weight([self.image_size//16 * self.image_size//16 * 64, 512], self.seed)
        self.b_fc1 = self.buil_bias([512])
        self.w_fc2 = self.buil_weight([512, self.num_labels], self.seed)
        self.b_fc2 = self.buil_bias([self.num_labels])
        
        
    # buil weight variable
    def buil_weight(self, shape, seed):
        initial = tf.truncated_normal(shape, stddev=0.1, seed=seed, dtype=tf.float32)
        return tf.Variable(initial)
    # buile bias variable
    def buil_bias(self, shape):
        initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
        return tf.Variable(initial)
    
    # buil CNN model
    def model(self,X, drop_out=False):
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
                
        # ********************** conv3 *********************************
        # transfer a 5*5*1 imagine into 64 sequence  
        with tf.name_scope('conv3'):
            with tf.name_scope('weight'):
                w_conv3 = self.w_conv3
            with tf.name_scope('bias'):
                b_conv3 = self.b_conv3
            with tf.name_scope('conv_and_pool'):
                conv3 = tf.nn.conv2d(pool2, w_conv3, strides=[1,1,1,1],padding='SAME')
                relu3 = tf.nn.relu(tf.nn.bias_add(conv3, b_conv3))
                pool3 = tf.nn.max_pool(relu3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        # ********************** conv4 *********************************
        # transfer a 5*5*1 imagine into 64 sequence  
        with tf.name_scope('conv4'):
            with tf.name_scope('weight'):
                w_conv4 = self.w_conv4
            with tf.name_scope('bias'):
                b_conv4 = self.b_conv4
            with tf.name_scope('conv_and_pool'):
                conv4 = tf.nn.conv2d(pool3, w_conv4, strides=[1,1,1,1],padding='SAME')
                relu4 = tf.nn.relu(tf.nn.bias_add(conv4, b_conv4))
                pool4 = tf.nn.max_pool(relu4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        # ********************* func1 layer *********************************
        # reshape the image from 8,8,64 into a flat (8*8*64),
        # and compute result
        with tf.name_scope('fc1'):
            with tf.name_scope('weight'):
                w_fc1 = self.w_fc1
            with tf.name_scope('bias'):
                b_fc1 = self.b_fc1
            with tf.name_scope('flatting'):    
                pool_shape = pool4.get_shape().as_list()
                reshape = tf.reshape(pool4, [pool_shape[0], pool_shape[1]*pool_shape[2]*pool_shape[3]])
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
            
# def main(_):
#     cnn = CNN(batch_size=64, image_size=64, num_channals=3, num_labels=2, seed=66478)
#     batch_data = train_data[0:64]
#     batch_labels = train_label[0:64]
#     X = tf.placeholder(tf.float32,shape=(64, 64, 64, 3))
#     y = tf.placeholder(tf.int64, shape=(64,))
#     logits,regularizers = cnn.model(X, drop_out=True)
#     loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)) + (5e-4*regularizers)
#     with tf.Session() as sess:
#         feed_dict = {X:batch_data,y:batch_labels}
#         tf.global_variables_initializer().run()
#         l = sess.run(loss,feed_dict=feed_dict)
#         print(l)
        
        
# if __name__ == "__main__":
#     tf.app.run()
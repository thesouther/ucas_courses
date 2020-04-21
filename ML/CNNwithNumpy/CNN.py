import math
import numpy as np

from layer.Convolution import Convolution
from layer.Relu import Relu
from layer.max_pool import max_pool
from layer.flatten import flatten
from layer.full_connection import full_connection
from layer.Softmax import Softmax

class CNN:
    def __init__(self,num_labels=10, batch_size=64, image_size=28, num_channels=1, seed=66478):
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.seed = seed
        self.num_labels=num_labels
        self.net_builder()
    
    # build network    
    def net_builder(self):
        self.conv1 = Convolution([self.batch_size, self.image_size, self.image_size, self.num_channels], k_size=5, k_num=6, strides=1, seed=2330, padding='VALID')
        self.relu1 = Relu(self.conv1.output_shape)
        self.pool1 = max_pool(self.relu1.output_shape, 2,2)
            
        self.conv2 = Convolution(self.pool1.output_shape, k_size=5, k_num=16, strides=1, seed=2330, padding='VALID')
        self.relu2 = Relu(self.conv2.output_shape)
        self.pool2 = max_pool(self.relu2.output_shape, 2,2)
            
        self.flat = flatten(self.pool2.output_shape)
        self.fc1 = full_connection(self.flat.output_shape,512,seed=self.seed)
        self.relu3 = Relu(self.fc1.output_shape)
            
        self.fc2 = full_connection(self.relu3.output_shape,10,seed=self.seed)
            
        self.softmax = Softmax(self.fc2.output_shape)
        
    def cal_forward(self,x):
        # forward for prediction
        conv1_out = self.conv1.forward(x)
        relu1_out = self.relu1.forward(conv1_out)
        pool1_out = self.pool1.forward(relu1_out)
        
        conv2_out = self.conv2.forward(pool1_out)
        relu2_out = self.relu2.forward(conv2_out)
        pool2_out = self.pool2.forward(relu2_out)
        
        flat_out = self.flat.forward(pool2_out)
        fc1_out = self.fc1.forward(flat_out)
        relu3_out = self.relu3.forward(fc1_out)
        
        fc2_out = self.fc2.forward(relu3_out)

        pred = self.softmax.prediction(fc2_out)
        return np.argmax(pred,axis=1)

    def fit(self, x, y, lr):
        # forward
        conv1_out = self.conv1.forward(x)
        relu1_out = self.relu1.forward(conv1_out)
        pool1_out = self.pool1.forward(relu1_out)
        
        conv2_out = self.conv2.forward(pool1_out)
        relu2_out = self.relu2.forward(conv2_out)
        pool2_out = self.pool2.forward(relu2_out)
        
        flat_out = self.flat.forward(pool2_out)
        fc1_out = self.fc1.forward(flat_out)
        relu3_out = self.relu3.forward(fc1_out)
        
        fc2_out = self.fc2.forward(relu3_out)

        # loss
        pred = self.softmax.prediction(fc2_out)
        loss = self.softmax.cal_loss(fc2_out, y)

        #backward
        loss_back = self.softmax.backward(label=y)
        fc2_back = self.fc2.backward(loss_back, lr=lr)
        relu3_back = self.relu3.backward(fc2_back)
        fc1_back = self.fc1.backward(relu3_back, lr=lr)
        flat_back = self.flat.backward(fc1_back)
        poo2_back = self.pool2.backward(flat_back)
        relu2_back = self.relu2.backward(poo2_back)
        conv2_back = self.conv2.backward(relu2_back)
        pool1_back = self.pool1.backward(conv2_back)
        relu1_back = self.relu1.backward(pool1_back)
        self.conv1.backward(relu1_back)
        return loss, np.argmax(pred,axis=1)

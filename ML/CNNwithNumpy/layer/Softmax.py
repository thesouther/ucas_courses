import math
import numpy as np

class Softmax:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.input_batch = input_shape[0]
        self.class_num = input_shape[1]
        
        self.softmax = np.zeros(self.input_shape)
        self.delta = np.zeros(self.input_shape)
        
    def cal_loss(self, x, label):
        self.prediction(x)
        loss = 0
        for i in range(self.input_batch):
            loss -= np.sum(np.log(self.softmax[i]) * label[i])
        loss /= self.input_batch
        return loss
    
    def prediction(self, x):
        for i in range(self.input_batch):
            pred_tmp = x[i, :] - np.max(x[i, :])
            pred_tmp = np.exp(pred_tmp)
            self.softmax[i] = pred_tmp/np.sum(pred_tmp)
        return self.softmax
    
    def backward(self, label):
        for i in range(self.input_batch):
            self.delta[i] = self.softmax[i] - label[i]
        return self.delta
    
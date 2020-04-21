import math
import numpy as np

class full_connection:
    def __init__(self, input_shape, output_channels, seed=2330):
        self.input_shape = input_shape
        self.input_batch = input_shape[0]
        self.input_length = input_shape[1]
        self.output_channels = output_channels
        
        self.seed = seed
        np.random.seed(self.seed)
        
        weights_scale = math.sqrt(self.input_length/2) 
        self.weights = np.random.standard_normal((self.input_length,self.output_channels)) / weights_scale
        self.bias = np.random.standard_normal(self.output_channels) / weights_scale
        
        self.output_shape = [self.input_batch, self.output_channels]
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        
    def forward(self, x):
        self.x = x
        y = np.dot(self.x, self.weights) +self.bias
        return y
    
    def backward(self, delta, lr=0.0001, weight_decay=0.0004):
        delta_back = np.dot(delta, self.weights.T)
        self.w_gradient = np.dot(self.x.T, delta)
        self.b_gradient = np.sum(delta, axis=0)
        
        self.weights = (1-weight_decay)*self.weights - lr*self.w_gradient
        self.bias = (1-weight_decay)*self.bias - lr*self.bias
        
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        return delta_back
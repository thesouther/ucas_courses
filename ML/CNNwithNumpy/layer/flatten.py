import math
import numpy as np

class flatten:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = [self.input_shape[0], self.input_shape[1] * self.input_shape[2] * self.input_shape[3]]
        
    def forward(self, x):
        y = x.reshape([self.input_shape[0], self.input_shape[1] * self.input_shape[2] * self.input_shape[3]])
        return y
    def backward(self, y):
        x = np.reshape(y, [self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3]])
        return x
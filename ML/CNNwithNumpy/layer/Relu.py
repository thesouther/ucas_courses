import math
import numpy as np

# rule Activator
class Relu:  
    def __init__(self, input_shape):
        self.delta = np.zeros(input_shape)
        self.input_shape = input_shape
        self.output_shape = self.input_shape
        
    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)
    
    def backward(self, delta):
        delta[self.x<0] = 0
        return delta
    
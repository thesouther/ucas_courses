import math
import numpy as np

class max_pool:
    def __init__(self, input_shape, k_size=2, strides=2):
        self.input_shape = input_shape
        self.k_size = k_size
        self.strides = strides
        self.output_shape = [input_shape[0], input_shape[1] // self.strides, input_shape[2] // self.strides, input_shape[3]]
    
    def forward(self, x):
        b, w, h, c = x.shape
        feature_w = w // self.strides
        feature = np.zeros((b, feature_w, feature_w, c))
        self.feature_mask = np.zeros((b, w, h, c))   # 记录最大池化时最大值的位置信息用于反向传播
        for bi in range(b):
            for ci in range(c):
                for i in range(0,h,self.strides):
                    for j in range(0,w, self.strides):
                        feature[bi, i//self.strides, j//self.strides, ci] = np.max(
                            x[bi,i:i+self.k_size,j:j+self.k_size,ci])
                        index = np.argmax(x[bi, i:i+self.k_size, j:j+self.k_size,ci])
                        self.feature_mask[bi, i+index//self.strides, j+index%self.strides, ci] = 1                    
        return feature

    def backward(self, delta):
        return np.repeat(np.repeat(delta, self.strides, axis=1), self.strides, axis=2) * self.feature_mask
import math
import numpy as np

class Convolution:
    def __init__(self, layer_shape, k_size=5, k_num=32, strides=1, seed=2330, padding='SAME'):
        self.input_shape = layer_shape
        self.input_batch = layer_shape[0]
        self.input_height = layer_shape[1]
        self.input_width = layer_shape[2]
        self.input_channels = layer_shape[3]
        
        self.seed = seed
        self.k_size = k_size
        self.strides = strides
        np.random.seed(self.seed)
        
        if (self.input_height-self.k_size) % self.strides != 0:
            print("input tensor height can\'t fit strides!")
        if (self.input_width-self.k_size) % self.strides != 0:
            print("input tensor width can\'t fit strides!")
        
        self.padding = padding
        if self.padding=='SAME':
            self.output_batch = self.input_batch
            self.output_height = self.input_height
            self.output_width = self.input_width
            self.output_channels = k_num
            self.delta = np.zeros((self.output_batch, self.output_height, self.output_width, self.output_channels))
        elif self.padding=='VALID':
            self.output_batch = self.input_batch
            self.output_height = (self.input_height-self.k_size) // self.strides + 1
            self.output_width = (self.input_width-self.k_size) // self.strides + 1
            self.output_channels = k_num
            self.delta = np.zeros((self.output_batch, self.output_height, self.output_width, self.output_channels))
        
        self.output_shape = self.delta.shape
        
        weights_scale = math.sqrt(k_size*k_size*self.output_channels/2) # init filter weights with dividing weights_scale 
        self.weights = np.random.standard_normal((k_size, k_size, self.input_channels, self.output_channels)) / weights_scale
        self.bias = np.random.standard_normal(self.output_channels) / weights_scale
        
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        
    def img2col(self, image, ksize, stride):
        # image is a 4d tensor([batchsize, width ,height, channel])
        image_col = []
        for i in range(0, image.shape[1] - ksize + 1, stride):
            for j in range(0, image.shape[2] - ksize + 1, stride):
                col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])
                image_col.append(col)
        image_col = np.array(image_col)

        return image_col
        
        
    def forward(self,X):
        col_weights = self.weights.reshape([-1, self.output_channels])
        # padding
        if self.padding == 'SAME':
            X = np.pad(X, 
                ((0,0), (self.k_size//2,self.k_size//2), (self.k_size//2, self.k_size//2),(0,0)),
                'constant', constant_values=(0,0))
#         print(X.shape)
        # convolution 
        self.col_image = []
        conv_out = np.zeros(self.delta.shape)
        for i in range(self.input_batch):
            img_i = X[i][np.newaxis, :]
            self.col_image_i = self.img2col(img_i, self.k_size, self.strides)
            conv_out[i] = np.reshape(np.dot(self.col_image_i, col_weights) + self.bias, self.delta[0].shape)
            self.col_image.append(self.col_image_i)
        self.col_image = np.array(self.col_image)
        return conv_out
        
    
    def backward(self, delta, lr=0.0001, weight_decay=0.0004):
        self.delta = delta
        col_delta = np.reshape(delta, [self.input_batch, -1, self.output_channels])
        
        for i in range(self.input_batch):
            self.w_gradient += np.dot(self.col_image[i].T, col_delta[i]).reshape(self.weights.shape)
        self.b_gradient += np.sum(col_delta, axis=(0,1))
        
        if self.padding == 'SAME':
            pad_delta = np.pad(self.delta, 
                                ((0,0), (self.k_size//2,self.k_size//2), (self.k_size//2, self.k_size//2),(0,0)),
                                'constant', constant_values=(0,0))
        else:
            pad_delta = np.pad(self.delta, 
                                ((0, 0), (self.k_size - 1, self.k_size - 1), (self.k_size - 1, self.k_size - 1), (0, 0)),
                                'constant', constant_values=0)
            
        flip_weights = np.flipud(np.fliplr(self.weights))
        flip_weights = flip_weights.swapaxes(2, 3)
        col_flip_weights = flip_weights.reshape([-1, self.input_channels])
        col_pad_delta = np.array([self.img2col(pad_delta[i][np.newaxis, :], self.k_size, self.strides) for i in range(self.input_batch)])
        delta_back = np.dot(col_pad_delta, col_flip_weights)
        delta_back = np.reshape(delta_back, self.input_shape)
        
        # update weights
        self.weights = (1-weight_decay)*self.weights - lr*self.w_gradient
        self.bias = (1-weight_decay)*self.bias - lr*self.bias
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        return delta_back

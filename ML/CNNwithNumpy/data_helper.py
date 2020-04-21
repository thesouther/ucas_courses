from __future__ import absolute_import,division,print_function

import gzip
import os
import sys
import numpy as np

# Extract the images into a 4D tensor [image index, y, x, channels].
# Values are rescaled from [0, 255] down to [-0.5, 0.5].
def extract_data(filename, num_images, image_size, num_channels,pixel_depth):
    
    print("Extracting data, ", filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(image_size*image_size*num_images*num_channels)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data-(pixel_depth/2.0)) / pixel_depth
        data = data.reshape(num_images, image_size, image_size, num_channels)
    return data

# Extract the labels into a vector of int64 label IDs.
def extract_labels(filename, num_images):
    print("Extracting labels,",filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1*num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

# get data
def load_train(validation_size,image_size, num_channels, pixel_depth):
    data_size = 60000
    file_directory = './data/'
    train_data_filename = file_directory + "train-images-idx3-ubyte.gz"
    train_labels_filename = file_directory + "train-labels-idx1-ubyte.gz"
    
    train_data = extract_data(train_data_filename, data_size, image_size, num_channels, pixel_depth)
    train_labels = extract_labels(train_labels_filename,data_size)

    validation_data = train_data[:validation_size, ...]
    validation_labels = train_labels[:validation_size]
    train_data = train_data[validation_size:, ...]
    train_labels = train_labels[validation_size:]
    return train_data,train_labels,validation_data,validation_labels

# get data
def load_test(image_size, num_channels, pixel_depth):
    data_size = 10000
    file_directory = './data/'
    test_data_filename = file_directory + "t10k-images-idx3-ubyte.gz"
    test_labels_filename = file_directory + "t10k-labels-idx1-ubyte.gz"
    
    test_data = extract_data(test_data_filename, data_size, image_size, num_channels, pixel_depth)
    test_labels = extract_labels(test_labels_filename,data_size)
    return test_data,test_labels



from __future__ import absolute_import,division,print_function

import gzip
import os
import sys
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# download data and return filepath
# if data has downloaded, set download=False
def download_data(file_directory, filename, download=False):
    if not tf.gfile.Exists(file_directory):
        tf.gfile.MakeDirs(file_directory)
    file_path = os.path.join(file_directory,filename)
    if download:
        source_url = 'http://yann.lecun.com/exdb/mnist/'
        file_path,_ =urllib.request.urlretrieve(source_url+filename, file_path)
        with tf.gfile.GFile(file_path) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes. ')
    return file_path

# Extract the images into a 4D tensor [image index, y, x, channels].
# Values are rescaled from [0, 255] down to [-0.5, 0.5].
def extract_data(filename, num_images, image_size, num_channels,pixel_depth):
    print("Extracting ", filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(image_size*image_size*num_images*num_channels)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data-(pixel_depth/2.0)) / pixel_depth
        data = data.reshape(num_images, image_size, image_size, num_channels)
    return data

# Extract the labels into a vector of int64 label IDs.
def extract_labels(filename, num_images):
    print("extracting labels,",filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1*num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

# def fake_data(num_images,image_size,num_channels):
#     """Generate a fake dataset that matches the dimensions of MNIST."""
#     data = np.ndarray(
#         shape=(num_images, image_size, image_size, num_channels),
#         dtype=np.float32)
#     labels=np.zeros(shape=(num_images,), dtype=np.int64)
#     for image in xrange(num_images):
#         label = image % 2
#         data[image, :, :, 0] = label-0.5
#         labels[image] = label
#     return data, labels




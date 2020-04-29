import cv2
import numpy as np
import glob
import os
from sklearn.utils import shuffle

CUR_DIR = os.getcwd()
train_cat_path = CUR_DIR+ '/data/train/cat/' 
train_dog_path = CUR_DIR+ '/data/train/dog/'
test_cat_path = CUR_DIR+ '/data/test/cat/'
test_dog_path = CUR_DIR+ '/data/test/dog/'

classes = ['dog', 'cat']

def load_data(data_path, category, image_resize):
    """ load images and resize to 'image_resize * image_resize', then append it to shape an array.
       data_path:    load_data path
       category:     dog or cat
       image_resize: resize image
    """
    files = glob.glob(data_path+'*')
    data=[]
    labels=[]
    
    for img in files:
        try:
            image = cv2.imread(img)
        except:
            print("img read error: ",img)
        try:
            image = cv2.resize(image, (image_resize, image_resize), 0, 0, cv2.INTER_LINEAR)
        except:
            print("img reseize error: ", img)
        image = image.astype(np.float32)
        image = np.multiply(image, 1.0/255.0)
        data.append(image)
#         label = np.zeros(len(classes))
#         label[classes.index(category)] = 1.0
        labels.append(classes.index(category))
    data = np.array(data)
    labels = np.array(labels).astype(np.int64)
    return data, labels
    
def extract_data(image_size, data_type='train'):
    """extract dog_images and cat_images accoding to data_type. 
       Then shuffle the data order using sklearn.utils.shuffle.
    """
    print("start extracting %s data" % data_type)
    if data_type=='train':
        dog_image, dog_labels = load_data(data_path=train_dog_path, category='dog', image_resize=image_size)
        cat_image, cat_labels = load_data(data_path=train_cat_path, category='cat', image_resize=image_size)
    elif data_type=='test':
        dog_image, dog_labels = load_data(data_path=test_dog_path, category='dog', image_resize=image_size)
        cat_image, cat_labels = load_data(data_path=test_cat_path, category='cat', image_resize=image_size)
    else:
        print("extract data type error: train/test!")
    # concatenate dog_image and cat_image.
    data_image = np.concatenate((dog_image,cat_image))
    # concatenate dog_labels and cat labels.
    data_label = np.concatenate((dog_labels, cat_labels))
    data_image,data_label = shuffle(data_image, data_label)
    
    return data_image,data_label

# def _test():
#     extract_data(64, data_type='test')
    
# _test()
import hashlib
import io
import logging
import os
import random
import re

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

def dict_to_tf_example(data, label_map_dict, image_subdirectory, ignore_difficult_instances=False):
    img_path = os.path.join(image_subdirectory, data['filename'])
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    for obj in data['object']:
        difficult_obj.append(int(0))

        xmin.append(float(obj['bndbox']['xmin']) / width)
        ymin.append(float(obj['bndbox']['ymin']) / height)
        xmax.append(float(obj['bndbox']['xmax']) / width)
        ymax.append(float(obj['bndbox']['ymax']) / height)

        class_name = obj['name']
        classes_text.append(class_name.encode('utf8'))
        classes.append(label_map_dict[class_name])
        truncated.append(int(0))
        poses.append('Unspecified'.encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example

def create_tf_record(output_filename, label_map_dict, trianval_xml_path, image_dir, examples):
    """从example中创建rfrecord格式数据"""
    writer = tf.python_io.TFRecordWriter(output_filename)
    for i,e in enumerate(examples):
        if i%100 ==0:
            print('create record for image %d of %d.' % (i, len(examples)))
        path = os.path.join(trianval_xml_path, e+'.xml')

        if not os.path.exists(path):
            print('file %s not find, continue.' % path)
            continue
        with tf.gfile.GFile(path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

        tf_example = dict_to_tf_example(data, label_map_dict, image_dir)
        writer.write(tf_example.SerializeToString())

    writer.close()

def main(_):
    from config import Config
    conf=Config()

    label_map_dict = label_map_util.get_label_map_dict(conf.label_map_path)

    print("load data")
    image_dir = conf.train_img_path
    trianval_xml_path = conf.trianval_xml_path
    example_path = conf.trianval_path
    example_list = dataset_util.read_examples_list(example_path)

    # 将数据随机分为训练集和验证集
    random.seed(42)
    random.shuffle(example_list)
    num_examples = len(example_list)
    num_train = int(conf.train_data_rate * num_examples)# 将训练数据分为训练集和验证集0.95：0.05
    train_examples = example_list[:num_train]
    val_examples = example_list[num_train:]
    print('train num: %d, val num: %d ' % (len(train_examples), len(val_examples)))

    train_record_path = conf.train_record_path
    val_record_path = conf.val_record_path
    create_tf_record(train_record_path, label_map_dict, trianval_xml_path, image_dir, train_examples)
    create_tf_record(val_record_path, label_map_dict, trianval_xml_path, image_dir, val_examples)

if __name__ == "__main__":
    tf.app.run()
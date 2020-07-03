## 基于tensorflow预训练模型的行人检测任务

本文中主要基于TensorFlow Object Detection API实现对行人的检测
___

### 一、准备工作
##### 1 环境搭建
本文使用环境：anaconda, tensorflow-1.11.0, python-3.6.5

##### 2 下载TensorFlow Object Detection API
- 地址：本文使用[r1.12.0](https://github.com/tensorflow/models/tree/r1.12.0)
- 下载模型，`git clone -b r1.12.0 https://github.com/tensorflow/models.git`
- 进入`models/research/`文件夹下，运行
  ```bash 
  python setup.py install
  ```
- 进入`models/research/slim/`文件夹下，运行
  ```bash 
  python setup.py build 
  python setup.py install
  ```
- 把`model/research/`目录下的`object_detection`和`slim`目录到为该实验创建的目录下。
- 其他环境依赖：
  - Protobuf 3.0.0
  - Python-tk
  - Pillow 1.0
  - lxml
  - tf Slim (which is included in the "tensorflow/models/- research/" checkout)
  - Jupyter notebook
  - Matplotlib
  - Tensorflow (>=1.9.0)
  - Cython
  - contextlib2
  - cocoapi

其中，cocoapi和Protobuf等可以按照[doc](https://github.com/tensorflow/models/blob/r1.12.0/research/object_detection/g3doc/installation.md)指引安装；
在这里为了方便，可以直接使用以下两个命令在anaconda中方便地安装：

```
pip install --user Cython
pip install --user contextlib2
pip install --user lxml
pip install --user pycocotools
pip install  tf_slim
pip install  absl-py
pip install pandas
pip install imageio
pip install protobuf
```

##### 3 下载Faster R-CNN预训练模型，
- 下载地址[detection_model_zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)。参考下载faster_rcnn_inception_v2_coco，也可以下载其他模型进行实验。主要需要如下文件：
    ```
    | - model.ckpt.meta
    | - model.ckpt.data-00000-of-00001
    | - model.ckpt.index
    | - pipeline.config
    ```
- 将`pipeline.config`放入根目录，修改配置内容。

##### 4 测试API
- **（重要）：进入`./models/research/`目录下，运行如下命令**
  ````
  protoc object_detection/protos/*.proto --python_out=.
  ````
- 将`object_detection`和`slim`加入环境变量(**每次都要重新运行**)：
  ```
  export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
  ```
- **测试**
  ```bash
  python object_detection/builders/model_builder_test.py
  ```
  不出错则API正确。

##### 5 下载数据集与预处理
- 数据集下载：[link](https://download.csdn.net/download/thenorther/12574200)
  该实验提供一个小型行人检测数据集`TownCentre`，该数据集包含一个视频`TownCentreXVID.avi`和标签文件`TownCentre-groundtruth.top`。
- 从avi视频中抽取图像帧，前4500帧图像作为训练数据，后3000帧作为测试数据。
- 从TownCentre-groundtruth.top提取前4500帧图像的行人位置信息，保存为xml文件。
- 将用于训练的所有帧的文件名写入data/annotation/trainval.txt文件中，即将0-4499写入到文件中，一行一个数字。
- 为行人指定一个id和标签，创建Dataset/annotation/label_map.pbtxt文件。
  
**上述四个功能可以通过运行如下命令完成**
```bash 
python data_helper.py
```
- 将上述获取的images和xml文件转为TFRecord文件
  ```bash
  python create_tf_record.py
  ```

### 二、运行：训练

##### 1 训练
```bash 
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

python object_detection/model_main.py \
    --pipeline_config_path='./pipeline.config' \
    --model_dir='./ckpt/' \
    --num_train_steps=20000 \
    --sample_1_of_n_eval_examples=1 \
    --alsologtostderr
```
##### 2 模型测试
```bash
python object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path=pipeline.config \
    --trained_checkpoint_prefix=ckpt/model.ckpt-20000 \
    --output_directory=pb_output
```
命令中第4行20000修改为你训练结束得到的最新的数字
##### 3 查看训练结果
```bash
tensorboard --logdir=ckpt
```
##### 4 测试，并生成测试数据的行人检测结果
```bash
python run_predict.py
```

运行过程中你可能会碰到各种问题，这里我总结了一些问题。如下。
### 三、 问题
##### 1. 问题1
```bash
TypeError: can't pickle dict_values objects
``` 
问题出现在`./object_detection/model_lib.py`下439行的`get_eval_metric_ops_for_evaluators`函数调用问题，

查看这个函数的定义可知，需要将第二个参数强转成list. 如下所示：

修改前：
```python
eval_metric_ops = eval_util.get_eval_metric_ops_for_evaluators(
          eval_config, lcategory_index.values(), eval_dict)
```
修改后：
```python 
eval_metric_ops = eval_util.get_eval_metric_ops_for_evaluators(
          eval_config, list(category_index.values()), eval_dict)
```
**reference**：[csdn blob](https://blog.csdn.net/qq_27882063/article/details/86094951)

##### 问题2
```
File "/home/ccong/anaconda3/envs/py36/lib/python3.6/site-packages/object_detection-0.1-py3.6.egg/object_detection/builders/anchor_generator_builder.py", line 21, in <module>
    from object_detection.protos import anchor_generator_pb2
ImportError: cannot import name 'anchor_generator_pb2'
```
如上等等的`object_detection`没法导入包，都是(二.4)部分没有做好。
- **（重要）：进入`./models/research/`目录下，运行如下命令**
  ````
  protoc object_detection/protos/*.proto --python_out=.
  ````
- 将`object_detection`和`slim`加入环境变量(**每次都要重新运行**)：
  ```
  export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
  ```
- **测试**
  ```bash
  python object_detection/builders/model_builder_test.py
  ```
  不出错则API正确。

##### 问题3
```
raise ValueError('First step cannot be zero.') 
ValueError: First step cannot be zero.
```
解决方法：删除pipeline.config中的下列代码

```
          ...
          schedule {
            step: 0
            learning_rate: 0.000199999994948
          }
          ...
```

### 四、文件结构
将文件组值为如下结构，有助于正确运行命令。

```
.
├─ckpt                          # 你自己继续训练的模型位置
├─object_detection              # 下载的API
├─slim                          # 下载的API
├─data                          # 数据文件夹
│  ├─val_data.record            # 验证集数据的record
│  ├─train_data.record          # 训练集数据的record
│  ├─annotations                # 行人信息
│  │  |─xmls
│  │  |─trainval.txt
│  │  └─label_map.pbtxt
│  ├─TownCentre                 # 原始视频数据
│  ├─test_images/               # 生成的测试图片
│  └─train_images/              # 生成的训练图片
├─pb_output                     # 测试模型时，export_inference_graph.py生成的模型文件
├─pretrained                    # 下载的预训练模型
├─result                        # 训练结果
│  |─output.txt
│  |─images.gif
│  └─images
├─config.py                     # 配置参数信息 
├─create_tf_record.py           #生成TFrecord
├─data_helper.py                # 提取图片、行人数据、生成gif等等
├─pipeline.config               # 模型管道配置文件
├─run_predict.py                # 预测脚本
└─README.md

```
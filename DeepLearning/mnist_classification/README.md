## 基于CNN的手写字符（MNIST）识别程序

本文基于LeNet5，实现手写字符识别程序，项目地址[https://github.com/thesouther/ucas_courses/tree/master/DeepLearning/mnist_classification](https://github.com/thesouther/ucas_courses/tree/master/DeepLearning/mnist_classification)

---
### 目录结构如下
本项目文件树如下结构
```
.
|-- data
|   |-- t10k-images-idx3-ubyte.gz
|   |-- t10k-labels-idx1-ubyte.gz
|   |-- train-images-idx3-ubyte.gz
|   `-- train-labels-idx1-ubyte.gz
|-- model
|   |-- checkpoint
|   |-- cnn_model.ckpt.data-00000-of-00001
|   |-- cnn_model.ckpt.index
|   `-- cnn_model.ckpt.meta
|-- logs
|   `-- events.out.tfevents.1586788885.bigdata-2
|-- data_helper.py
|-- CNN.py
|-- train.py
|-- test.py
|-- run.sh
`-- README.md
```

其中`data`文件存放mnist数据集，`model`文件夹保存训练好的数据，`logs`文件夹存放tensorflow中的图结构和loss曲线等数据，`CNN_logs.py`文件为CNN模型文件，`data_helper.py`文件用于载入训练数据和测试数据，`test.py`使用训练好的模型进行测试，`train.py`用于使用训练数据训练模型，`run.sh`为本人提供的运行脚本。

### 运行

**训练**
```
bash run.sh train
```
**测试**
```
bash run.sh test
```
**训练参数说明：**
运行 `python train.py -h`，显示参数说明：

```
usage: train.py [-h] [-e EPOCHS] [-b BATCH_SIZE] [-f EVAL_FREQUENCY]
                [-o {momentum,adam}] [-s] [-g] [-v VALIDATION_SIZE]

CNN by CC

optional arguments:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        number of epochs for train
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        batch size for training
  -f EVAL_FREQUENCY, --eval-frequency EVAL_FREQUENCY
                        evaluation frequency
  -o {momentum,adam}, --optimizer {momentum,adam}
                        optimizer, you can select from [momentum, adam]
  -s, --save            path to save the final model
  -g, --use-gpu         select do or don't use GPU for training
  -v VALIDATION_SIZE, --validation-size VALIDATION_SIZE
                        validation data size
```


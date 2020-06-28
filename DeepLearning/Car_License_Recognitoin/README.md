## 基于CNN的车牌识别程序

本项目基于Pytorch编写的基于CNN的车牌识别程序。

----

### 命令

首先请将老是提供的数据集`tf_car_license_dataset`中的`test, train, validation`文件夹提取到`dataset`目录下。

使用`python .\run.py -h`查看命令帮助。
```
usage: run.py [-h] [--train] [--test] -t {provinces,area,letters} [-s]

car license recognition use CNN

optional arguments:
  -h, --help            show this help message and exit
  --train               train the model
  --test                test the model
  -t {provinces,area,letters}, --type {provinces,area,letters}
                        train type, select from [provinces,area,letters]
  -s, --save            save the final model
```

### 训练与测试
使用如下命令运行训练代码
```
python .\run.py --train -t 'provinces' --save --show
python .\run.py --train -t 'area' --save --show
python .\run.py --train -t 'letters' --save --show
```

测试模型：
```
python .\run.py --test -t 'provinces' >> ./result/test.txt
python .\run.py --test -t 'area' >> ./result/test.txt
python .\run.py --test -t 'letters' >> ./result/test.txt
```

### 目录结构如下
本文目录结构如下：
```
.
|-- model/          # 存放模型文件
|-- result          # 存放结果
|   |-- provinces.png
|   |-- area.png
|   `-- letters.png
|-- data_helper.py  # 预处理数据集，并载入训练数据和测试数据
|-- model.py        # CNN模型
|-- run.py          # 训练和测试脚本
|-- config.py       # 保存各种超参数信息
`-- README.md
```

### 模型结构

在`show_model_info()`函数中写出了查看CNN模型信息的代码，可以运行查看：
```bash
python model.py
```

**结果如下**

```
CNN(
  (conv1): Sequential(
    (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv2): Sequential(
    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv3): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (fc1): Linear(in_features=2048, out_features=512, bias=True)
  (dropout): Dropout(p=0.6, inplace=False)
  (fc2): Linear(in_features=512, out_features=6, bias=True)
)

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             320
       BatchNorm2d-2           [-1, 32, 32, 32]              64
              ReLU-3           [-1, 32, 32, 32]               0
         MaxPool2d-4           [-1, 32, 16, 16]               0
            Conv2d-5           [-1, 64, 16, 16]          18,496
       BatchNorm2d-6           [-1, 64, 16, 16]             128
              ReLU-7           [-1, 64, 16, 16]               0
         MaxPool2d-8             [-1, 64, 8, 8]               0
            Conv2d-9            [-1, 128, 8, 8]          73,856
      BatchNorm2d-10            [-1, 128, 8, 8]             256
             ReLU-11            [-1, 128, 8, 8]               0
        MaxPool2d-12            [-1, 128, 4, 4]               0
           Linear-13                  [-1, 512]       1,049,088
          Dropout-14                  [-1, 512]               0
           Linear-15                    [-1, 6]           3,078
================================================================
Total params: 1,145,286
Trainable params: 1,145,286
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.43
Params size (MB): 4.37
Estimated Total Size (MB): 5.80
----------------------------------------------------------------
```

### 超参数

在`config.py`文件中查看和修改。


## 实验结果

在`./result/`文件夹下查看训练过程和测试结果。

```
---------- test:  provinces ----------
Extracting test provinces data and labels 
true label: 闽,	 predict lable: 闽
true label: 苏,	 predict lable: 苏
true label: 京,	 predict lable: 京
true label: 吉,	 predict lable: 京
---------- test:  area ----------
Extracting test area data and labels 
true label: O,	 predict lable: O
true label: Q,	 predict lable: Q
true label: K,	 predict lable: K
true label: B,	 predict lable: B
true label: E,	 predict lable: E
true label: I,	 predict lable: I
---------- test:  letters ----------
Extracting test letters data and labels 
true label: 2,	 predict lable: 2
true label: 0,	 predict lable: 0
true label: 8,	 predict lable: 8
true label: B,	 predict lable: B
true label: 7,	 predict lable: 7
true label: 9,	 predict lable: 9

```
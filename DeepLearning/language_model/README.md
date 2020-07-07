## 基于Text-CNN的电影评论情感分类

本文主要实现基于Pytorch的神经网络语言模型

----

### 运行

使用如下命令运行训练代码
```
python main.py > ./result/train.txt
```

模型loss曲线在`./result/`文件夹下。

### 查看结果

在`./result/`文件夹下查看训练过程结果。

### 目录结构如下
本文目录结构如下：
```
.
|-- model
|-- result
|-- data_helper.py
|-- model.py
|-- main.py
|-- config.py
`-- README.md
```

- `model`文件夹存放模型checkpoints；
- `result`文件夹保存训练和测试结果；
- `data_helper.py`文件用于预处理数据集，并载入训练数据和测试数据；
- `model.py`文件为RNN模型脚本；
- `main.py`用于使用训练数据训练模型；
- `config.py`用于保存各种超参数信息。

### 超参数
在`config.py`脚本中，您可以方便查看和修改。
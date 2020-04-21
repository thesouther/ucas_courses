## 基于python基础库实现卷积神经网络LeNet5

从聪，  201928015029010，  congcong19@mails.ucas.ac.cn

**本程序实现为ipynb和py脚本两种，ipynb文件更加易读，所以建议阅读`完整版CNN.ipynb`文件**

--
#### 简介
本程序只基于Python的numpy库，实现cnn网络Lenet5，并用于在MNIST数据集上进行手写字符识别。

本程序实现了convolution、relu、max_pooling、fc、softmax等层级的前向和后向算法，并在mnist数据集的测试集上在3个epoch就可以实现**98%** 以上的的准确率。训练时间大概一个小时。

#### 运行与查看

1. 为了方便，本人整理了了notebook形式的文件，记录了运行过程中的中间结果，你可以查看`完整版CNN.ipynb`文件，或者直接从我的[github](https://github.com/thesouther/ucas_courses/tree/master/ML/CNNwithNumpy)方便地查看效果。
2. python文件。进行训练和测试，运行
   ```
   python run.py
   ```

#### 文件目录说明
文件目录树如下：
```
.
|-- data
|   |-- t10k-images-idx3-ubyte.gz   # mnist测试数据
|   |-- t10k-labels-idx1-ubyte.gz   # mnist测试标签
|   |-- train-images-idx3-ubyte.gz  # mnist训练数据
|   `-- train-labels-idx1-ubyte.gz  # mnist训练标签
|-- layer
|   |-- Convolution.py              # 卷积层
|   |-- Relu.py                     # Relu激活函数
|   |-- Softmax.py                  # softmax层
|   |-- flatten.py                  # 将二维数据展开成一维
|   |-- full_connection.py          #全连接层
|   `-- max_pool.py                 # 池化层
|-- data_helper.py                  # 载入数据
|-- CNN.py                          # CNN模型定义类
|-- run.py                          # 训练和测试代码
|-- 完整版CNN.ipynb                 # 完整的CNN代码，可以方便查看
|-- README.md

```

#### 运行环境
```
python = 3.6.10
numpy = 1.16.0
```

### Reference

- [https://zhuanlan.zhihu.com/c_162633442](https://zhuanlan.zhihu.com/c_162633442)
- [https://www.cnblogs.com/qxcheng/p/11729773.html](https://www.cnblogs.com/qxcheng/p/11729773.html)
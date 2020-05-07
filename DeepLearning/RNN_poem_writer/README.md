## 基于RNN的中文自动写诗程序

本项目基于tensorflow编写的RNN神经网络，实现自动写唐诗。

----

### 运行

使用如下命令运行训练代码
```
python train.py > ./result/train.txt
```

测试模型：
```
python test.py >> ./result/test.txt
```

使用`tensorboard --logdir=logs`查看网络结构和loss曲线。

### 查看结果

在`./result/`文件夹下查看训练过程和测试结果。

### 目录结构如下
本文目录结构如下：
```
.
|-- data
|   `-- tang.npz
|-- logs
|   `-- events.out.....bigdata-2
|-- model
|   `-- checkpoints
|-- result
|   `-- train.txt
|-- data_helper.py
|-- model.py
|-- train.py
|-- test.py
`-- README.md
```

- `data`文件存放预处理过的唐诗数据集，包括
    ```
    data: 诗词数据，将诗词中的字转化为其在字典中的序号表示。
    ix2word: 序号到字的映射
    word2ix: 字到序号的映射
    ```
- `logs`文件夹存放tensorflow中的图结构和loss曲线等数据；
- `model`文件夹存放模型checkpoints；
- `result`文件夹保存训练和测试结果；
- `data_helper.py`文件用于预处理数据集，并载入训练数据和测试数据；
- `model.py`文件为RNN模型脚本；
- `train.py`用于使用训练数据训练模型。
- `test.py`用于测试模型生成的古诗。

### 超参数
修改如下超参数进行训练
```
NUM_EPOCH = 10
BATCH_SIZE = 64
len_vector = 125
embedding_dim = 256
n_neurons = embedding_dim
n_layers = 3
lr = 0.001
keep_prob=0.8
eval_frequence = 100
model='lstm' # lstm, gru, rnn
```
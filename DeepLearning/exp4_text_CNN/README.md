## 基于Text-CNN的电影评论情感分类

本项目基于tensorflow编写的基于Text-CNN的电影评论情感分类程序。

本文实现了两种Text-CNN结构，其主要不同在于，一种使用了单一的filter结构（详情请查看`model.py`）；

另一种使用了多种不同filter的组合方式，详细结构请查看`model2.py`。

想要查看两种不同结构的效果，请修改超参数：`config.py`文件下，`select_model`参数。

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
|-- logs
|   `-- events.out.....bigdata-2
|-- model
|   `-- checkpoints
|-- result
|   |-- train.txt
|   |-- train2.txt
|   |-- test.txt
|   `-- test2.txt
|-- data_helper.py
|-- model.py
|-- model2.py
|-- train.py
|-- test.py
|-- config.py
`-- README.md
```


- `logs`文件夹存放tensorflow中的图结构和loss曲线等数据；
- `model`文件夹存放模型checkpoints；
- `result`文件夹保存训练和测试结果；
- `data_helper.py`文件用于预处理数据集，并载入训练数据和测试数据；
- `model.py`文件为Text-CNN模型脚本；
- `model2.py`文件为使用多中卷积核的Text-CNN模型脚本；
- `train.py`用于使用训练数据训练模型；
- `test.py`用于测试模型；
- `config.py`用于保存各种超参数信息。

### 超参数
修改如下超参数进行训练
```
self.select_model = 'model2'           #选择模型，两个选择，['model', 'model2']

if self.select_model == 'model2':    
    self.kernel_size = [2,3,4,5] # 卷积核的尺寸；nlp任务中通常选择2,3,4,5
    self.kernel_classes = 4      # 卷积核的种类；nlp任务中通常选择2,3,4,5
else: 
    self.kernel_size = 4         # 卷积核的尺寸；nlp任务中通常选择2,3,4,5
self.seed=66478
self.update_w2v = True           # 是否在训练中更新w2v
self.vocab_size = 58954          # 词汇量，与word2id中的词汇量一致
self.n_class = 2                 # 分类数：分别为pos和neg
self.max_sen_len = 75            # 句子最大长度
self.embedding_dim = 50          # 词向量维度
self.batch_size = 100            # 批处理尺寸
self.n_hidden = 256              # 隐藏层节点数
self.n_epoch = 10                # 训练迭代周期，即遍历整个训练样本的次数
self.opt = 'adam'                # 训练优化器：adam或者adadelta
self.learning_rate = 0.001       # 学习率；若opt=‘adadelta'，则不需要定义学习率
self.drop_keep_prob = 0.5        # dropout层，参数keep的比例
self.input_channel = 1
self.num_filters = 256           # 卷积层filter的数量
self.print_per_batch = 10      # 训练过程中,每100词batch迭代，打印训练信息
self.save = True     
self.save_dir = './model/'       # 训练模型保存的地址
self.train_path = '../Dataset/train.txt'
self.val_path = '../Dataset/validation.txt'
self.test_path = '../Dataset/test.txt'
self.word2id_path = '../Dataset/word_to_id.txt'
self.pre_word2vec_path = '../Dataset/wiki_word2vec_50.bin'
self.corpus_word2vec_path = '../Dataset/corpus_word2vec.txt'
self.classes = {'0': 0, '1': 1}  # 分类标签

```
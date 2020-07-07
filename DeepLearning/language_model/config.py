# -*- coding: UTF-8 -*-
import torch
class Config:
    def __init__(self):
        self.train = True
        self.save_model = True
        self.show_loss = False

        self.select_model = 'lstm'           #选择模型，两个选择，['lstm', 'gru']
        self.batch_size = 20            # 批处理尺寸
        self.num_epochs = 10
        self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

        self.seq_length = 30            # 序列长度
        self.embed_size = 128           # 词嵌入长度
        self.hidden_size = 512
        self.num_layers = 3
        self.opt = 'adam'                # 训练优化器：adam或者adadelta
        self.learning_rate = 0.002       # 学习率；若opt=‘adadelta'，则不需要定义学习率

        self.test_num_samples = 10
        
        self.print_per_batch = 100      # 训练过程中,每100词batch迭代，打印训练信息  
        self.model_path = './model/RNN.model'       # 训练模型保存的地址
        self.train_path = './data/ptb.train.txt'
        self.valid_path = './data/ptb.valid.txt'
        self.test_path = './data/ptb.test.txt'
        self.result_fig_path = './result/loss.png'



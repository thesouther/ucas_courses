# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
 
# RNN语言模型
class RNN(nn.Module): #RNNLM类继承nn.Module类
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        #嵌入层 one-hot形式(vocab_size,1) -> (embed_size,1)
        self.embed = nn.Embedding(vocab_size, embed_size)
        #LSTM单元/循环单元
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        #输出层的全联接操作  
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, h):
        # 词嵌入
        x = self.embed(x)
        
        # LSTM前向运算
        out,(h,c) = self.lstm(x,h)
 
        # 每个时间步骤上LSTM单元都会有一个输出，batch_size个样本并行计算(每个样本/序列长度一致)  out (batch_size,sequence_length,hidden_size)
        # 把LSTM的输出结果变更为(batch_size*sequence_length, hidden_size)的维度
        out = out.reshape(out.size(0)*out.size(1),out.size(2))
        # 全连接
        out = self.linear(out) #(batch_size*sequence_length, hidden_size)->(batch_size*sequence_length, vacab_size)
        return out,(h,c)


# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states] 

def test():
    import torch
    import torch.nn as nn
    from torch.nn.utils import clip_grad_norm_
    import numpy as np

    from data_helper import Corpus
    from config import Config
    conf = Config()

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    C = Corpus(conf)
    word2id = C.word2id
    vocab_size = len(word2id)
    print("vocab_size", vocab_size)

    # 导入数据
    print("extracting data... ")
    train_data, valid_data, test_data = C.build_dataset(conf)

    train_size = train_data.size(1)
    
    # 实例化模型
    model = RNN(vocab_size, conf.embed_size, conf.hidden_size, conf.num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)
    states = (torch.zeros(conf.num_layers, conf.batch_size, conf.hidden_size).to(device),
              torch.zeros(conf.num_layers, conf.batch_size, conf.hidden_size).to(device))
    for i in range(2):
        batch_x = train_data[:, i:(i+conf.seq_length)].to(device)
        batch_y = train_data[:, (i+1) : ((i+1+conf.seq_length)%train_size)].to(device)

        # 前传
        states = detach(states)
        outputs,states = model(batch_x, states)
        print("outputs.size()",outputs.size())
        print(batch_y.reshape(-1).size())
        loss = criterion(outputs, batch_y.reshape(-1))

if __name__ == "__main__":
    test()
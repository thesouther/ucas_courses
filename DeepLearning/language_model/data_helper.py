# -*- coding: UTF-8 -*-
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def save_results(loss_count, result_fig_path, show=False):
    plt.figure()
    plt.plot(loss_count,color="red",label='Loss')
    plt.title("train loss curve")
    plt.ylabel("loss")      
    plt.xlabel("steps (x 100)")

    plt.savefig(result_fig_path)
    if show:
        plt.show()

def eval_model(valid_data, conf, states, model):
    """生成预测值"""
    valid_size = valid_data.size(1)
    seq_length = conf.seq_length
    batch_size = conf.batch_size
    device = conf.device
    perp, count = 0,0
    criterion = nn.CrossEntropyLoss()
    for i in range(0, valid_size-seq_length, seq_length):
        batch_x = valid_data[:, i:(i+seq_length)].to(device)
        batch_y = valid_data[:, (i+1) : ((i+1+seq_length)%valid_size)].to(device)
        outputs,states = model(batch_x, states)

        loss = criterion(outputs, batch_y.reshape(-1))
        perp += np.exp(loss.item())
        count += 1
    return perp / count

class Corpus:
    def __init__(self,conf):
        self.conf = conf
        self.word2id = {}
        self.id2word = {}
        self.idx = 0
        # 使用训练数据和验证数据构造字典
        self.word2id_from_file(conf.train_path)
        # self.word2id_from_file(conf.valid_path)

    def word2id_from_file(self,path):
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                words = line.strip().split() + ['<EOS>']
                for w in words:
                    if not w in self.word2id:
                        self.word2id[w] = self.idx
                        self.id2word[self.idx] = w
                        self.idx += 1

    def build_from_word2id(self, path, word2id):
        with open(path, 'r') as f:
            data = f.read().replace('\n', '<EOS>')
            data_ids = [word2id[w] for w in data if w in word2id]
        return data_ids

    def build_dataset(self,conf):     
        # 使用字典构造训练集、验证集、测试集
        train_ids = self.build_from_word2id(conf.train_path, self.word2id)
        test_ids = self.build_from_word2id(conf.test_path, self.word2id)
        valid_ids = self.build_from_word2id(conf.valid_path, self.word2id)
        train_ids = torch.from_numpy(np.array(train_ids)).type(torch.LongTensor)
        test_ids = torch.from_numpy(np.array(test_ids)).type(torch.LongTensor)
        valid_ids = torch.from_numpy(np.array(valid_ids)).type(torch.LongTensor)

        # 按照batch size 划分数据
        train_batchs = train_ids.size(0) // conf.batch_size
        test_batchs = test_ids.size(0) // conf.batch_size
        valid_batchs = valid_ids.size(0) // conf.batch_size
        train_ids = train_ids[: train_batchs* conf.batch_size].view(conf.batch_size, -1)
        test_ids = test_ids[: test_batchs* conf.batch_size].view(conf.batch_size, -1)
        valid_ids = valid_ids[: valid_batchs* conf.batch_size].view(conf.batch_size, -1)
    
        return train_ids, valid_ids, test_ids

if __name__ == "__main__":
    from config import Config
    conf = Config()
    C = Corpus(conf)
    train_ids, valid_ids, test_ids = C.build_dataset(conf)

    word2id, dict_len = C.word2id, len(C.word2id)
    id2word = C.id2word
    print(dict_len)

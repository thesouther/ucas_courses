# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import numpy as np
import random

from model import RNN
from data_helper import Corpus, save_results, eval_model
from config import Config
conf = Config()

device = conf.device

# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states] 

def main():
    print("extracting corpus... ")
    # 导入词典
    C = Corpus(conf)
    word2id, vocab_size = C.word2id, len(C.word2id)
    id2word = C.id2word

    # 导入数据
    print("extracting data... ")
    train_data, valid_data, test_data = C.build_dataset(conf)

    train_size = train_data.size(1)
    
    # 实例化模型
    model = RNN(vocab_size, conf.embed_size, conf.hidden_size, conf.num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)

    # 训练开始
    loss_count = []
    for epoch in range(conf.num_epochs):
        print("="*20,"epoch: %d" % epoch, "="*20)
        states = (torch.zeros(conf.num_layers, conf.batch_size, conf.hidden_size).to(device),
              torch.zeros(conf.num_layers, conf.batch_size, conf.hidden_size).to(device))
        for i in range(0, train_size-conf.seq_length, conf.seq_length):
            batch_x = train_data[:, i:(i+conf.seq_length)].to(device)
            batch_y = train_data[:, (i+1) : ((i+1+conf.seq_length)%train_size)].to(device)

            # 前传
            states = detach(states)
            outputs,states = model(batch_x, states)
            loss = criterion(outputs, batch_y.reshape(-1))

            # BP 
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            step = (i+1) // conf.seq_length
            if step % conf.print_per_batch == 0:
                loss_count.append(loss.item())
                valid_acc = eval_model(valid_data, conf, states, model)
                print("step: %d,\t Loss: %.3f,\t train Perplextity: %.3f,\t validation Perplextity: %.3f." % (
                    step, loss.item(), np.exp(loss.item()), valid_acc*100
                ))
    # 展示loss曲线
    save_results(loss_count, conf.result_fig_path, show=conf.show_loss)
    # 保存模型
    if conf.save_model:
        print("save model: %s" % conf.model_path)
        torch.save(model, conf.model_path)

if __name__ == "__main__":
    main()
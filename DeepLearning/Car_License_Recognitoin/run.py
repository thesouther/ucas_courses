# -*- coding: UTF-8 -*-
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset,DataLoader

from model import CNN
from data_helper import load_train, load_test, eval_in_batches, show_results

from config import Config
conf = Config()

device = torch.device(conf.device if torch.cuda.is_available() else 'cpu')

def run_train(args):
    # 设置训练的类型，是provinces, area, letters 
    conf.set_train_params(args.type)

    # 载入数据
    train_loader, val_loader, train_size = load_train(conf)
    conf.train_size = train_size

    # 实例化模型和loss函数、优化器
    model = CNN(conf).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    if conf.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)
    else:
        optimizer = torch.optim.Adadelta(model.parameters())

    # 开始训练
    loss_count, acc_count = [],[]
    for epoch in range(conf.n_epoch):
        print("="*15,"epoch: ",str(epoch), "="*20)
        for i, (x, y) in enumerate(train_loader):
            x = x.unsqueeze(1)
            batch_x = Variable(x)
            batch_y = Variable(y)

            output = model(batch_x)
            loss = criterion(output, batch_y)

            optimizer.zero_grad() 
            loss.backward()

            optimizer.step()

            if i % conf.print_per_batch ==0:
                loss_count.append(loss)
                print('train step: {} [{}/{} ({:.0f}%)]'.format(i, 
                    i*len(batch_x), 
                    conf.train_size, 
                    100.*i/len(train_loader))
                )
                batch_acc = 100.0 * (output.argmax(1) == batch_y).float().sum().item() / len(batch_x)
                print("\t minibatch loss: %.3f,\t acc: %.1f%%" % (loss.item(), batch_acc))
                acc_count.append(batch_acc)
                print("\t validation accuracy: %.1f%%" % (eval_in_batches(val_loader, model)*100))
    
    if args.show:
        show_results(conf.train_type, loss_count,acc_count, conf.result_fig_path, show=False)

    # 保存模型
    if args.save:
        print("save model: %s" % conf.model_path)
        torch.save(model, conf.model_path)

def run_test(args):
    # 设置训练的类型，是provinces, area, letters 
    conf.set_train_params(args.type)
    print("-"*10, "test: ", args.type, "-"*10)
    model = torch.load(conf.model_path)

    test_data,test_labels = load_test(conf)
    test_loader = DataLoader(dataset = test_data, batch_size = 1, 
                            shuffle = False)

    for i, x in enumerate(test_loader):
        x = x.unsqueeze(1)
        test_x = Variable(x)
        output = model(test_x)
        pre_label = output.argmax(1).item()
        print('true label: %s,\t predict lable: %s' %(test_labels[i], conf.classes[pre_label]))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="car license recognition use CNN")
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--test', action='store_true',
                        help='test the model')
    parser.add_argument('-t','--type', type=str, default='provinces', choices=['provinces','area','letters'],
                        required=True, help="train type, select from [provinces,area,letters]")
    parser.add_argument('-s','--save', action='store_true',
                        help='save the final model')
    parser.add_argument('--show', action='store_true',
                        help='show loss,acc curves')
    args = parser.parse_args()
    if args.train:
        run_train(args)
    if args.test:
        run_test(args)

# -*- coding: UTF-8 -*-
import numpy as np
import gzip
import os
import glob
from torch.autograd import Variable
import cv2
import torch
from torch.utils.data import TensorDataset,DataLoader
import matplotlib.pyplot as plt

def eval_in_batches(dataloader, model):
    """生成预测值"""
    count, correct = 0, 0
    for _, (x, y) in enumerate(dataloader):
        x = x.unsqueeze(1)
        batch_x = Variable(x)
        batch_y = Variable(y)
        output = model(batch_x)
        correct += (output.argmax(1) == batch_y).float().sum().item()
        count += len(batch_x)
    return correct/count

def load_train(conf):
    print("Extracting train %s data and labels " % conf.train_type)
    train_data, train_labels = extract_data(conf.train_path , conf.bisic_range, conf.num_classes, conf.image_shape)
    val_data, val_labels = extract_data(conf.validation_path, conf.bisic_range, conf.num_classes, conf.image_shape)
    train_size = len(train_labels)

    # 加载训练用的数据
    train_dataset = TensorDataset(torch.from_numpy(train_data).type(torch.float), 
                                torch.from_numpy(train_labels).type(torch.long))
    train_loader = DataLoader(dataset = train_dataset, batch_size = conf.batch_size, 
                              shuffle = True)
    # 加载验证集的数据
    val_dataset = TensorDataset(torch.from_numpy(val_data).type(torch.float), 
                                torch.from_numpy(val_labels).type(torch.long))
    val_loader = DataLoader(dataset = val_dataset, batch_size = conf.batch_size, 
                              shuffle = False)
    return train_loader, val_loader, train_size

def load_test(conf):
    print("Extracting test %s data and labels " % conf.train_type)
    test_data = []
    test_labels = []
    for file in conf.test_fig_names.keys():
        img = conf.test_path + file
        if ' ' in img:
            continue
        try:
            image = cv2.imread(img)
        except:
            print("img read error: ",img)
        try:
            image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        except:
            print("img change to gray error: ", img)
        try:
            image = cv2.resize(image, (conf.image_shape,conf.image_shape), cv2.INTER_LINEAR)
        except:
            print("img resize error: ", img)
        image = image.astype(np.float32)
        image = np.multiply(image, 1.0/255.0)
        test_data.append(image)
        test_labels.append(conf.test_fig_names[file])

    test_data = np.array(test_data)
    test_data = torch.from_numpy(test_data)
    test_labels = np.array(test_labels).astype(np.str)

    return test_data,test_labels

def extract_data(data_dir, bisic_range, num_classes, img_resize):
    data = []
    labels = []
    for i in range(num_classes):
        file_path = data_dir + str(i+bisic_range)+'/'
        files = glob.glob(file_path+'*')
        for img in files:
            if ' ' in img:
                continue
            try:
                image = cv2.imread(img)
            except:
                print("img read error: ",img)
            try:
                image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            except:
                print("img change to gray error: ", img)
            try:
                image = cv2.resize(image, (img_resize,img_resize), cv2.INTER_LINEAR)
            except:
                print("img resize error: ", img)
            image = image.astype(np.float32)
            # print(image.shape)
            image = np.multiply(image, 1.0/255.0)
            data.append(image)
            labels.append(i)

    data = np.array(data)
    labels = np.array(labels).astype(np.int64)
    return data, labels

def show_results(train_type, loss, acc, result_fig_path, show=True):
    plt.figure()
    ax1 = plt.subplot(211)
    ax1.plot(loss,color="red",label='Loss')
    ax1.set_title("train: " + train_type + ", loss curve")
    ax1.set_ylabel("loss")      
    ax1.set_xlabel("steps (x 10)")

    ax2 = plt.subplot(212)
    ax2.plot(acc,color="blue",label='accuracy')
    ax2.set_title("train: " + train_type + ", accuracy curve")
    ax2.set_ylabel("accuracy/(%)")      
    ax2.set_xlabel("steps (x 10)")

    plt.subplots_adjust(wspace =0, hspace =0.5)
    plt.savefig(result_fig_path)
    if show:
        plt.show()

def test():
    from config import Config
    conf = Config()
    conf.set_train_params('provinces')
    # print(conf.num_classes)
    # train_loader, val_loader, train_size = load_train(conf)
    # print(train_size)
    # print(train_dataloader)

    # show_results(conf.train_type, [4.3, 3.2, 4.2, 2, 1, 0.1], [0.1,0.2,0.3,0.4,0.5,0.6])

    test_data,test_labels = load_test(conf)
    print(test_data.size())
    print(test_labels)
    # for i, (x,y) in enumerate(test_loader):
    #     test_x = Variable(x)
    #     test_y = Variable(y)
    #     # print(test_x)

if __name__ == "__main__":
    test()
    
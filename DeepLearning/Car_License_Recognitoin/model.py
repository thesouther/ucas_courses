# -*- coding: UTF-8 -*-

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import numpy as np

class CNN(torch.nn.Module):
    def __init__(self,conf):
        super(CNN,self).__init__()
        self.conf = conf
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=conf.conv1_input_channels,
                out_channels=conf.conv1_num_filters,
                kernel_size=conf.conv1_kerne_size,
                stride=conf.conv1_kerne_stride,
                padding=conf.conv1_padding),
            torch.nn.BatchNorm2d(conf.conv1_num_filters),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=conf.pool1_size, 
                stride=conf.pool1_strides
            )
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=conf.conv2_input_channels,
                out_channels=conf.conv2_num_filters,
                kernel_size=conf.conv2_kerne_size,
                stride=conf.conv2_kerne_stride,
                padding=conf.conv2_padding),
            torch.nn.BatchNorm2d(conf.conv2_num_filters),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=conf.pool2_size, 
                stride=conf.pool2_strides
            )
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=conf.conv3_input_channels,
                out_channels=conf.conv3_num_filters,
                kernel_size=conf.conv3_kerne_size,
                stride=conf.conv3_kerne_stride,
                padding=conf.conv3_padding),
            torch.nn.BatchNorm2d(conf.conv3_num_filters),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=conf.pool3_size, 
                stride=conf.pool3_strides
            )
        )
        self.fc1 = torch.nn.Linear(conf.fc1_input_channels, conf.fc1_output_channels)
        self.dropout = torch.nn.Dropout(conf.drop_keep_prob)
        self.fc2 = torch.nn.Linear(conf.fc2_input_channels, conf.fc2_output_channels)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc1(x.view(x.size(0),-1))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def show_model_info():
    from config import Config
    conf = Config()
    conf.set_train_params('provinces')
    device = torch.device(conf.device if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model = CNN(conf).to(device)
    print(model)

    from torchsummary import summary
    summary(model, input_size=(conf.conv1_input_channels, conf.image_shape, conf.image_shape))

if __name__ == "__main__":
    show_model_info()



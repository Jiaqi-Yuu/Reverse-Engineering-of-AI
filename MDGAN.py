'''
    file name: model.py
    create time: 3/26/2022
    modify time: 3/26/2022 19:36
'''

from colorsys import rgb_to_hls
import imp
import torch
import torch.nn as nn 
import torch.nn.functional as F
import pdb
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import Resize
import torchvision.models as models
import numpy 
import math

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)

class DNN(nn.Module):
    def __init__(self, dim_list):
        super(DNN, self).__init__()
        self.MLP = nn.ModuleList()
        for i in range(len(dim_list) - 1):
            self.MLP.append(nn.Linear(dim_list[i], dim_list[i + 1]))

    def forward(self, x):
        for i in range(len(self.MLP)):
            x = self.MLP[i](x)
            if i != len(self.MLP) - 1:
                x = F.relu(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.ModuleList([
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ])
    def forward(self, x):       # x表示28*28的图片
        for i in range(len(self.main)):
            x = self.main[i](x)
        return x

class CNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNN,self).__init__()
        self.conv_input = nn.Conv2d(
            in_channels=input_dim,
            out_channels=3,
            kernel_size=3,
            stride=1
        )

        self.conv_middle = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            stride=1
        )

        self.conv_final = nn.Conv2d(
            in_channels=3,
            out_channels=output_dim,
            kernel_size=3,
            stride=1
        )
        self.convolution = nn.Sequential(
            self.conv_input,
            nn.ReLU(),
            self.conv_middle,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv_middle,
            nn.ReLU(),
            self.conv_final,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
    def forward(self, x):
        x = self.convolution(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        return x

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.DNN_query = CNN(input_dim=3, output_dim=1)
        self.DNN_output = DNN([10, 10, 16])
        self.ENC_max_mi = DNN([700, 500, 128])
        self.ENC_min_mi = DNN([700, 500, 128])

    def forward(self, tm_outputs):
        fea = tm_outputs
        fea_max_mi = self.ENC_max_mi(fea)
        fea_min_mi = self.ENC_min_mi(fea)
        return fea_max_mi, fea_min_mi

class Classification(nn.Module):
    def __init__(self, attr_dict):
        super(Classification, self).__init__()
        attr_value_keys = sorted(attr_dict)
        attr_value_list = []
        for key in attr_value_keys:
            attr_value_list.append(len(attr_dict[key]))
        self.hidden_dim_list_list = [[128, 128, k] for k in attr_value_list]
        self.attr_num = len(attr_dict)

        self.meta = nn.ModuleList([MetaModel(self.hidden_dim_list_list[i]) for i in range(self.attr_num)])

    def forward(self, x, ret_last_hid=False):
        if ret_last_hid:
            x_last_hidden_list = [self.meta[i](x, ret_last_hid=True) for i in range(self.attr_num)]
            return x_last_hidden_list
        else:
            x_list = [self.meta[i](x) for i in range(self.attr_num)]
            prob_list = [F.softmax(x, dim=1) for x in x_list]
            return prob_list


        
class MetaModel(nn.Module):
    def __init__(self, hidden_dim_list):
        super(MetaModel, self).__init__()
        print(hidden_dim_list)
        self.MLP = nn.ModuleList()
        for i in range(len(hidden_dim_list) - 1):
            self.MLP.append(nn.Linear(hidden_dim_list[i], hidden_dim_list[i + 1]))
        print(self.MLP)
    def forward(self, x, ret_last_hid=False):
        x_last_hid = 0
        for i in range(len(self.MLP)):
            x = self.MLP[i](x)
            if i == len(self.MLP) - 2:
                x_last_hid = x
            if i != len(self.MLP) - 1:
                # print(x)
                x = F.relu(x)
        if ret_last_hid:
            return x_last_hid
        else:
            return x


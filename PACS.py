import torch.nn as nn
import torch.nn.functional as F

class convblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, bn="None"):
        super(convblock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        
        self.relu = nn.ReLU(inplace=True)
        self.bn = bn
        if bn == "normal":
            self.bn_layer = nn.BatchNorm2d(num_features=in_channels)
    def forward(self, x):
        x = self.conv(x)
        if self.bn == "normal":
            x = self.bn_layer(x)
        x = self.relu(x)
        

        return x


class linearblock(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dropout='none'):
        super(linearblock, self).__init__()
        self.conv = nn.Linear(in_features, out_features, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(inplace=True)
        self.dropoutoption = dropout

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        if self.dropoutoption == 'normal':
            x = self.dropout(x)
        return x


class mlpblock(nn.Module):
    def __init__(self, inputdim, outputdim, nlayer=2, hiddendim=10, activation='ReLU'):
        super(mlpblock, self).__init__()
        self.nlayer = nlayer
        self.hiddendim = hiddendim
        self.inputdim = inputdim
        self.outputdim = outputdim

        if activation == 'ReLU':
            self.act = F.relu
        else:
            raise NotImplementedError

        self.fc1 = nn.Linear(self.inputdim, self.hiddendim)
        fc_iter = []
        for n in range(self.nlayer - 2):
            fc_iter.append(linearblock(self.hiddendim, self.hiddendim))
        self.fc_iter = nn.Sequential(*fc_iter)
        self.fc_final = nn.Linear(self.hiddendim, self.outputdim)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc_iter(x)
        x = self.fc_final(x)
        return x

class return_act(nn.Module):
    def __init__(self, act):
        super(return_act, self).__init__()
        if act == 'relu':
            self.actfunc = nn.ReLU()
        elif act == 'elu':
            self.actfunc = nn.ELU()
        elif act == 'prelu':
            self.actfunc = nn.PReLU()
        elif act == 'tanh':
            self.actfunc = nn.Tanh()
        else:
            raise ValueError('Activation type should be one of {relu, elu, prelu, tanh}.')

    def forward(self, x):
        return self.actfunc(x)

class mnet(nn.Module):
    def __init__(self, control, channel_num=3):
        super(mnet, self).__init__()
        self.control = control
        self.act = return_act(control['net']['act'])

        self.ks = self.control['net']['ks']
        self.conv1 = nn.Conv2d(channel_num, 10, kernel_size=self.ks)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=self.ks)

        if 'normal' in self.control['net']['bn']:
            self.bn1 = nn.BatchNorm2d(num_features=10)
            self.bn2 = nn.BatchNorm2d(num_features=20)
        
        if 'max' in self.control['net']['pool']:
            stride = int(self.control['net']['pool'].split('_')[1])
            self.pool = lambda a: F.max_pool2d(a, stride)
            poolfactor = 2
        else:
            self.pool = lambda a: a
            poolfactor = 1

        conv_iter = []
        for n in range(self.control['net']['n_conv'] - 2):
            conv_iter.append(convblock(20, 20, self.ks, padding=int((self.ks - 1) / 2), bn=self.control['net']['bn']))
        self.conv_iter = nn.Sequential(*conv_iter)

        self.fcfeatdim = int((
                                     (
                                             (
                                                     (
                                                             (32 - self.ks + 1) // poolfactor
                                                     ) - self.ks + 1) // poolfactor
                                     ) ** 2
                             ) * 20)

        self.fc1 = nn.Linear(self.fcfeatdim, 50)
        fc_iter = []
        for n in range(self.control['net']['n_fc'] - 2):
            fc_iter.append(linearblock(50, 50, self.control['net']['drop']))
        self.fc_iter = nn.Sequential(*fc_iter)
        self.fc_final = nn.Linear(50, 7)

    def forward(self, x):
        if 'normal' in self.control['net']['bn']:
            x = self.act.forward(self.pool(self.bn1(self.conv1(x))))
            x = self.act.forward(self.pool(self.bn2(self.conv2(x))))
        else:
            x = self.act.forward(self.pool(self.conv1(x)))
            x = self.act.forward(self.pool(self.conv2(x)))
        x = self.conv_iter(x)
        x = x.view(-1, self.fcfeatdim)
        x = self.act.forward(self.fc1(x))
        if self.control['net']['drop'] == 'normal':
            x = F.dropout(x, training=self.training)
        x = self.fc_iter(x)
        x = self.fc_final(x)
        return F.log_softmax(x)
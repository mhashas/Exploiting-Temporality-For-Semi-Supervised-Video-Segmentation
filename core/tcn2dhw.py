"""
Code adapted from: https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size, :-self.chomp_size].contiguous()


class TemporalBlock2DHW(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, apply_weight_norm=False):
        super(TemporalBlock2DHW, self).__init__()
        self.conv1 = nn.Conv2d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp2d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        '''
        self.conv2 = nn.Conv2d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp2d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        '''

        if apply_weight_norm:
            self.conv1 = weight_norm(self.conv1)
            #self.conv2 = weight_norm(self.conv2)

        self.relu = nn.ReLU()
        self.init_weights()
        self.kernel_size = kernel_size

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.conv1.weight.data, 0.02)
        #torch.nn.init.xavier_normal_(self.conv2.weight.data, 0.02)


    def forward(self, x):
        out = self.conv1(x)

        if self.kernel_size > 1:
            out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        '''
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        '''
        res = x
        return self.relu(out + res)


class TemporalConvNet2DHW(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet2DHW, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock2DHW(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)

        return x
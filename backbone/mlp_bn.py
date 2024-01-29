# -*- coding: utf-8 -*-
# @Author  : jacob xu
# @Time    : 2023/8/1 21:17
# @File    : mlp_bn.py
# @Software: PyCharm
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class mlp(nn.Module):
    # infeatures: 输入的特征维度,就是表格的列数
    def __init__(self, in_features, out_features, hidden_sizes, drop_p):
        super(mlp, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes
        self.drop_p = drop_p

        # self.encoder1 = nn.Sequential(
        #     nn.Linear(in_features, hidden_sizes, bias=True),
        #     nn.LeakyReLU(negative_slope=0.05),
        #     nn.Dropout(drop_p),
        #     nn.Linear(hidden_sizes, 2, bias=True)
        # )
        self.encoder1 = nn.Sequential(
            nn.Linear(in_features, hidden_sizes, bias=True),
            # nn.BatchNorm1d(hidden_sizes),  # Batch Normalization layer added
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(hidden_sizes, out_features, bias=True)
        )

        # Initialize the weights using Kaiming Initialization
        for layer in self.encoder1:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')


    # def forward(self, inputs):
    #     embeddings = self.encoder1(inputs)
    #     return embeddings
    def forward(self, inputs, fast_weights=None):
        if fast_weights is None:
            # Use current model parameters if fast_weights is not provided
            embeddings = self.encoder1(inputs)
        else:
            # Use fast_weights for fast adaptation
            x = inputs
            fast_weight_idx = 0
            for layer in self.encoder1:
                if isinstance(layer, nn.Linear):
                    weight = fast_weights[fast_weight_idx]
                    bias = fast_weights[fast_weight_idx + 1]
                    x = F.linear(x, weight=weight, bias=bias)
                    # x = F.linear(x, weight=weight.t(), bias=bias)
                    fast_weight_idx += 2
                else:
                    x = layer(x)
            embeddings = x
        return embeddings
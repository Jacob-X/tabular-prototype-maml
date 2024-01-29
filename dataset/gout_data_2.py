# -*- coding: utf-8 -*-
# @Author  : jacob xu
# @Time    : 2023/8/26 21:44
# @File    : gout_data_2.py
# @Software: PyCharm
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


class goutDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.data = self.data.drop(["Unnamed: 0"], axis=1)  # 删除Unnamed列
        self.data = self.data.drop(["eid"], axis=1)  # 删除eid列
        self.data = self.data.fillna(self.data.median())  # 用中位数填充缺失值
        self.labels = self.data.iloc[:, -9:].values  # 获取最后9列作为标签
        self.features = self.data.iloc[:, :-9].values  # 获取除最后9列以外的特征数据
        self.scaler = MinMaxScaler()  # 创建归一化器
        self.normalized_features = self.scaler.fit_transform(self.features)  # 对特征数据进行归一化

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.normalized_features[idx]
        labels = self.labels[idx]
        features = torch.Tensor(features)
        labels = torch.Tensor(labels)
        return features, labels
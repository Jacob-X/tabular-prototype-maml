# -*- coding: utf-8 -*-
# @Author  : jacob xu
# @Time    : 2023/7/13 20:17
# @File    : gout_data.py
# @Software: PyCharm

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class GoutDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.labels = self.data.iloc[:, -9:].values  # 获取最后10列作为标签
        self.features = self.data.iloc[:, 1:-9].values  # 获取除最后10列以外的特征数据
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

if __name__ == '__main__':

    dataset = GoutDataset("/home/xutianxin/Pycharm_code/multi-labels-prototypical-networks/dataset/gout_resampled_data_2.csv")
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    print(len(dataloader))

# -*- coding: utf-8 -*-
# @Author  : jacob xu
# @Time    : 2023/8/1 16:24
# @File    : evaluate_model.py
# @Software: PyCharm

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score,precision_score, accuracy_score
import copy
import torch.utils.data as data
import csv
import torch.nn as nn

def evaluate(model, test_dataloader, criterion, epoch, args,device,filename):
    # test_dataloader是把所有数据放在一个batch里面的
    for i, (batch_features, batch_labels) in enumerate(test_dataloader):

        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        split_ratio = [2/3, 1/3]
        split_sizes = [int(len(batch_features) * ratio) for ratio in split_ratio]
        split_dataset = data.random_split(list(zip(batch_features, batch_labels)), split_sizes)

        # 获取划分后的数据集
        train_data = split_dataset[0]
        val_data = split_dataset[1]

        # 解压缩数据集
        support_features, support_labels = zip(*train_data)
        query_features, query_labels = zip(*val_data)
        # for column in range(len(batch_labels[0])):

        support_features = torch.stack(support_features, dim=0)
        support_labels = torch.stack(support_labels, dim=0)
        query_features = torch.stack(query_features, dim=0)
        query_labels = torch.stack(query_labels, dim=0)

        eval_metrics_list = []
        acc_list = []

        for column in range(len(batch_labels[0])):

            positive_col = ["ckd", "heart_failure", "hypertension", "myocardial_infaraction", "nephrolithiasis",
                            "obesity", "stroke", "t2DM", "gout", ]

            task_model = copy.deepcopy(model).to(device)
            column_values = support_labels[:, column]

            feature_embedding = task_model(support_features)
            # 根据当前列的值为 1 和 0 的索引，从 train_x 中提取对应的数据
            train_positive = feature_embedding[column_values == 1]
            train_negative = feature_embedding[column_values == 0]

            # 从 train_positive 中随机采样 5 个数据
            pos_sampler = data.RandomSampler(train_positive, replacement=True, num_samples=5)
            neg_sampler = data.RandomSampler(train_negative, replacement=True, num_samples=5)

            # 创建一个数据加载器，用于加载采样后的数据
            pos_dataloader = data.DataLoader(train_positive, batch_size=5, sampler=pos_sampler)
            neg_data_loader = data.DataLoader(train_negative, batch_size=5, sampler=neg_sampler)

            # 遍历数据加载器，获取采样后的数据
            pos_sampled_data = []
            neg_sampled_data = []
            for batch in pos_dataloader:
                pos_sampled_data.append(batch)
            for batch in neg_data_loader:
                neg_sampled_data.append(batch)

            # 这里相当于得到了正负两类原型
            train_positive_mean = torch.mean(pos_sampled_data[0], dim=0)
            train_negative_mean = torch.mean(neg_sampled_data[0], dim=0)

            query_features_embed = task_model(query_features)

            ce_loss = []
            for embed in query_features_embed:
                pos_euclidean_distance = torch.sqrt(pow(embed - train_positive_mean, 2).sum())
                neg_euclidean_distance = torch.sqrt(pow(embed - train_negative_mean, 2).sum())

                # 计算softmax概率
                pos_softmax_probs = -torch.exp(pos_euclidean_distance) / (
                        - torch.exp(pos_euclidean_distance) - torch.exp(neg_euclidean_distance))
                neg_softmax_probs = -torch.exp(neg_euclidean_distance) / (
                        - torch.exp(pos_euclidean_distance) - torch.exp(neg_euclidean_distance))

                softmax_probs = torch.stack([pos_softmax_probs, neg_softmax_probs], dim=0)
                # 计算交叉熵损失
                ce_loss.append(softmax_probs)


            # 这里是对模型进行快速迭代
            ce_loss_list = torch.stack(ce_loss, dim=0)
            query_values = query_labels[:, column]

            ce_loss_value = criterion(ce_loss_list, query_values.long())
            grad = torch.autograd.grad(ce_loss_value, task_model.parameters())
            fast_weights = list(map(lambda p: p[1] - args.update_lr * p[0], zip(grad, task_model.parameters())))

            for k in range(1, args.update_step-1):
                ce_loss_2 = []
                ce_loss_list_2 = []
                sop_embed = task_model(query_features, fast_weights)
                for embed in sop_embed:
                    pos_euclidean_distance = torch.sqrt(pow(embed - train_positive_mean, 2).sum())
                    neg_euclidean_distance = torch.sqrt(pow(embed - train_negative_mean, 2).sum())
                    # 计算softmax概率
                    pos_softmax_probs = -torch.exp(pos_euclidean_distance) / (
                            - torch.exp(pos_euclidean_distance) - torch.exp(neg_euclidean_distance))
                    neg_softmax_probs = -torch.exp(neg_euclidean_distance) / (
                            - torch.exp(pos_euclidean_distance) - torch.exp(neg_euclidean_distance))
                    softmax_probs = torch.stack([pos_softmax_probs, neg_softmax_probs], dim=0)
                    # 计算交叉熵损失
                    ce_loss_2.append(softmax_probs)
                ce_loss_list_2 = torch.stack(ce_loss_2, dim=0).requires_grad_(True)
                query_values = query_labels[:, column]
                query_values = query_values.requires_grad_(True)
                ce_loss_value_2 = criterion(ce_loss_list_2, query_values.long())
                grad = torch.autograd.grad(ce_loss_value_2, fast_weights)
                fast_weights = list(map(lambda p: p[1] - args.update_lr * p[0], zip(grad, fast_weights)))

            # 最后一次对模型性能进行验证

            with torch.no_grad():
                ce_loss_last = []
                ce_loss_list_last = []
                predicted_class_list = []
                predicted_val_result = []
                sop_embed = task_model(query_features, fast_weights)
                for embed in sop_embed:
                    pos_euclidean_distance = torch.sqrt(pow(embed - train_positive_mean, 2).sum())
                    neg_euclidean_distance = torch.sqrt(pow(embed - train_negative_mean, 2).sum())
                    # 计算softmax概率
                    pos_softmax_probs = -torch.exp(pos_euclidean_distance) / (
                            - torch.exp(pos_euclidean_distance) - torch.exp(neg_euclidean_distance))
                    neg_softmax_probs = -torch.exp(neg_euclidean_distance) / (
                            - torch.exp(pos_euclidean_distance) - torch.exp(neg_euclidean_distance))
                    softmax_probs = torch.stack([pos_softmax_probs, neg_softmax_probs], dim=0)
                    # 计算交叉熵损失
                    ce_loss_last.append(softmax_probs)

                    # # 找到具有最高概率的类别索引
                    predicted_class = torch.argmax(softmax_probs)
                    predicted_class_list.append(predicted_class)


                ce_loss_list_last = torch.stack(ce_loss_last, dim=0)
                query_values_last = query_labels[:, column]
                predicted_result = torch.stack(predicted_class_list, dim=0)

                cross_entropy_loss = criterion(ce_loss_list_last, query_values_last.long())

                # mse
                # mse_loss = nn.MSELoss()
                # mse_value = mse_loss(predicted_result, query_values_last)

                # 计算准确率
                true_labels = query_values_last.cpu().numpy()  # 真实标签
                predicted_labels = predicted_result.cpu().numpy()  # 模型预测结果

                # 计算F1分数
                f1 = f1_score(true_labels, predicted_labels)

                # 计算AUC
                auc = roc_auc_score(true_labels, predicted_labels)

                accuracy = accuracy_score(true_labels, predicted_labels)
                precision = precision_score(true_labels, predicted_labels)

                acc_list.append(accuracy)

                eval_metrics_list.append([positive_col[column], cross_entropy_loss.item(),f1, auc, accuracy, precision])


        column_names = ["disease","loss","f1_score", "auc", "accuracy", "precision"]

        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            for eval_metrics in eval_metrics_list:
                writer.writerow(eval_metrics)

        print("Evaluation metrics saved to CSV:", filename)

        return np.mean(acc_list),acc_list
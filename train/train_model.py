# -*- coding: utf-8 -*-
# @Author  : jacob xu
# @Time    : 2023/8/1 16:24
# @File    : train_model.py
# @Software: PyCharm

import torch
import torch.utils.data as data
import copy
import csv

def train(model, train_dataloader, val_dataloader,optimizer,criterion, epoch, args,device,filename):

    model.train()

    for i,(batch_features, batch_labels) in enumerate(train_dataloader):

        # 将批次数据和标签转移到GPU（如果可用）
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        # 计算划分的比例
        if len(batch_features) != 2048:
            break

        split_ratio = [0.75, 0.25]
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

        all_ce_loss_list = []
        # 对每个标签都要做不同的处理，每个标签都使用不同的模型，然后计算梯度，最后累加梯度进行更新
        for column in range(len(support_labels[0])):

            print(f"Column: {column}")
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
                            torch.exp(pos_euclidean_distance) + torch.exp(neg_euclidean_distance))
                neg_softmax_probs = -torch.exp(neg_euclidean_distance) / (
                            torch.exp(pos_euclidean_distance) + torch.exp(neg_euclidean_distance))

                softmax_probs = torch.stack([pos_softmax_probs, neg_softmax_probs], dim=0)
                # 计算交叉熵损失
                ce_loss.append(softmax_probs)

            ce_loss_list = torch.stack(ce_loss, dim=0)
            query_values = query_labels[:, column]

            ce_loss_value = criterion(ce_loss_list, query_values.long())
            grad = torch.autograd.grad(ce_loss_value, task_model.parameters())
            fast_weights = list(map(lambda p: p[1] - args.update_lr * p[0], zip(grad, task_model.parameters())))

            for k in range(1, args.update_step):
                ce_loss_2 = []
                ce_loss_list_2 = []
                sop_embed = task_model(query_features, fast_weights)
                for embed in sop_embed:
                    pos_euclidean_distance = torch.sqrt(pow(embed - train_positive_mean, 2).sum())
                    neg_euclidean_distance = torch.sqrt(pow(embed - train_negative_mean, 2).sum())
                    # 计算softmax概率
                    pos_softmax_probs = -torch.exp(pos_euclidean_distance) / (
                             torch.exp(pos_euclidean_distance) + torch.exp(neg_euclidean_distance))
                    neg_softmax_probs = -torch.exp(neg_euclidean_distance) / (
                             torch.exp(pos_euclidean_distance) + torch.exp(neg_euclidean_distance))
                    softmax_probs = torch.stack([pos_softmax_probs, neg_softmax_probs], dim=0)
                    # 计算交叉熵损失
                    ce_loss_2.append(softmax_probs)
                ce_loss_list_2 = torch.stack(ce_loss_2, dim=0)
                query_values = query_labels[:, column]
                ce_loss_value_2 = criterion(ce_loss_list_2, query_values.long())
                grad = torch.autograd.grad(ce_loss_value_2, fast_weights)
                fast_weights = list(map(lambda p: p[1] - args.update_lr * p[0], zip(grad, fast_weights)))

            # 在验证集上进行模型评估
                for val_features, val_labels in val_dataloader:

                    val_features = val_features.to(device)
                    val_labels = val_labels.to(device)
                    predicted_logits_result = []

                    task_model.zero_grad()
                    val_embed = task_model(val_features)

                    val_embed.grad = None
                    # 对每个样本都要计算一次损失
                    for embed in val_embed:
                        val_ce_loss = []

                        pos_euclidean_distance = torch.sqrt(pow(embed - train_positive_mean, 2).sum())
                        neg_euclidean_distance = torch.sqrt(pow(embed - train_negative_mean, 2).sum())

                        # 计算softmax概率
                        pos_softmax_probs = -torch.exp(pos_euclidean_distance) / (
                                 torch.exp(pos_euclidean_distance) + torch.exp(neg_euclidean_distance))
                        neg_softmax_probs = -torch.exp(neg_euclidean_distance) / (
                                 torch.exp(pos_euclidean_distance) + torch.exp(neg_euclidean_distance))
                        softmax_probs = torch.stack([pos_softmax_probs, neg_softmax_probs])

                        val_ce_loss.append(softmax_probs)

                        predicted_logits = torch.stack(val_ce_loss, dim=0)
                        predicted_logits_result.append(predicted_logits)


                    predicted_logits_tensor = torch.stack(predicted_logits_result,dim=0)

                    positive_col = ["ckd", "heart_failure", "hypertension", "myocardial_infaraction", "nephrolithiasis",
                                    "obesity", "stroke", "t2DM", "gout"]

                    print("dataloader num: ", i, "disease name:", positive_col[column])

                    val_values = val_labels[:, column]
                    predicted_logits_values_2 = predicted_logits_tensor[:, 0].detach().cpu().numpy()
                    predicted_logits_values_2 = torch.from_numpy(predicted_logits_values_2).to(device)
                    predicted_logits_values_2.requires_grad_(True)

                    ce_loss_value_3 = criterion(predicted_logits_values_2, val_values.long())
                    # fast_weights = list(map(lambda p: p[1] - args.update_lr * p[0], zip(grad, task_model.parameters())))
                    print("disease name:", positive_col[column], 'CrossEntropyLoss:', ce_loss_value_3)
                    # 把当前任务的最后一步的损失加入到列表中
                    if k == args.update_step - 1:
                        all_ce_loss_list.append(ce_loss_value_3)

        model.zero_grad()
        # for loss in ce_loss_list:
        #     loss.backward()

        optimizer.zero_grad()
        avg_loss = torch.mean(torch.stack(all_ce_loss_list))
        print("avg_loss:", avg_loss)
        print("++++++++++" * 10)
        # 反向传播
        avg_loss.backward()
        optimizer.step()


        write_loss_list = []
        for loss_item in all_ce_loss_list:
            write_loss_list.append(loss_item.item())

        write_loss_list.append(avg_loss.item())

        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(write_loss_list)
        file.close()


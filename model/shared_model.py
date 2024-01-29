# -*- coding: utf-8 -*-
# @Author  : jacob xu
# @Time    : 2023/7/17 22:22
# @File    : shared_model.py
# @Software: PyCharm

from sklearn.model_selection import KFold
import numpy as np
import torch
from torch.utils.data import random_split,Subset,DataLoader
from gout_data import GoutDataset
import random, math
import torch.nn as nn
from model.model import mlp
from sklearn.metrics import f1_score, roc_auc_score,precision_score, accuracy_score
from utils import arg_parse
import logging
import shutil
import copy
import torch.utils.data as data
import csv
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(filename="training_1w.log",
                    format='%(asctime)s - %(message)s',
                    level=logging.INFO)

def log_metrics(epoch,num,name, mse_value, ce_loss_value, f1, auc, accuracy, precision):
    # logging.info(f"Epoch {epoch}: ")
    # logging.info(f"MSE: {mse_value}")
    # logging.info(f"Dataloader Num: {num}: ")
    # logging.info(f"Disease name: {name}")
    # logging.info(f"Cross entropy loss: {ce_loss_value}")
    # logging.info(f"F1 score: {f1}")
    # logging.info(f"AUC: {auc}")
    # logging.info(f"Accuracy: {accuracy}")
    # logging.info(f"Precision: {precision}")
    logging.info(
        f"Epoch {epoch}: MSE: {mse_value}, Dataloader Num: {num}, Disease name: {name}, "
        f"Cross entropy loss: {ce_loss_value}, F1 score: {f1}, AUC: {auc},"
        f"Accuracy: {accuracy}, Precision: {precision}")


def save_checkpoint(state, is_best, filename="checkpoint_3.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best_3.pth.tar")

# Fix random seed for reproducibility
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train(model, train_dataloader, val_dataloader,optimizer,criterion, epoch, args):

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
                            - torch.exp(pos_euclidean_distance) - torch.exp(neg_euclidean_distance))
                neg_softmax_probs = -torch.exp(neg_euclidean_distance) / (
                            - torch.exp(pos_euclidean_distance) - torch.exp(neg_euclidean_distance))

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
                            - torch.exp(pos_euclidean_distance) - torch.exp(neg_euclidean_distance))
                    neg_softmax_probs = -torch.exp(neg_euclidean_distance) / (
                            - torch.exp(pos_euclidean_distance) - torch.exp(neg_euclidean_distance))
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
                                - torch.exp(pos_euclidean_distance) - torch.exp(neg_euclidean_distance))
                        neg_softmax_probs = -torch.exp(neg_euclidean_distance) / (
                                - torch.exp(pos_euclidean_distance) - torch.exp(neg_euclidean_distance))
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

        with open('train_loss_values_3.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(write_loss_list)
        file.close()


def evaluate(model, test_dataloader, criterion, epoch, args):
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

                # 交叉熵损失
                # ce_loss_value_last = criterion(ce_loss_list_last, query_values_last.long())

                # # mse
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

                # eval_metrics_list.append([positive_col[column],ce_loss_value_last.item(), mse_value.item(), f1, auc, accuracy, precision])

                eval_metrics_list.append([positive_col[column], f1, auc, accuracy, precision])

        csv_file_path = "evaluation_metrics_3.csv"
        column_names = ["disease","f1_score", "auc", "accuracy", "precision"]

        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            # writer.writerow(["Column"] + column_names)  # Header row
            for eval_metrics in eval_metrics_list:
                writer.writerow(eval_metrics)

        print("Evaluation metrics saved to CSV:", csv_file_path)

        return np.mean(acc_list),acc_list

def mian(args):
    # 定义k-fold参数和epoch参数
    k = 5  # k-fold的折数
    num_epochs = 100  # 训练的轮数

    best_acc1 = 0

    # 创建数据集
    dataset = GoutDataset("/home/xutianxin/Pycharm_code/multi-labels-prototypical-networks/dataset/gout_resampled_data_2.csv")

    # 创建模型
    model = mlp(in_features=77, out_features=256, hidden_sizes=512, drop_p=0.5).to(device)

    criterion = nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # 获取数据集长度
    num_samples = len(dataset)

    # 划分训练集和测试集
    train_ratio = 0.8  # 训练集比例
    train_size = int(train_ratio * num_samples)
    test_size = num_samples - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建k-fold交叉验证对象
    kfold = KFold(n_splits=k)

    acc_list = []
    csv_file_path = "evaluation_metrics_3.csv"
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["disease","ce_loss", "mse", "f1_score", "auc", "accuracy", "precision"])# Header row
        file.close()

    positive_col = ["ckd", "heart_failure", "hypertension", "myocardial_infaraction", "nephrolithiasis",
                    "obesity", "stroke", "t2DM", "gout", ]

    with open('train_loss_values_3.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(positive_col+["avg_loss"])  # Header row
        file.close()


    for epoch in tqdm(range(num_epochs)):

    # 进行k-fold交叉验证
        for fold, (train_indices, val_indices) in enumerate(kfold.split(train_dataset)):
            # 根据fold划分训练集和验证集
            train_subset = Subset(train_dataset, train_indices)
            val_subset = Subset(train_dataset, val_indices)

            # 创建训练集和验证集的DataLoader
            train_dataloader = DataLoader(train_subset, batch_size=2048, shuffle=True)
            val_dataloader = DataLoader(val_subset, batch_size=len(val_subset), shuffle=False)

        # 在每个fold内进行epoch训练迭代

            # 调整学习率
            adjust_learning_rate(optimizer, epoch, args)

            train(model, train_dataloader, val_dataloader,optimizer,criterion, epoch, args)

            print("Fold:", fold+1)
            print("Epoch:", epoch+1)
            print("Train samples:", len(train_subset))
            print("Validation samples:", len(val_subset))
            print("------------------------------")

    # 在测试集上进行评估
        test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        average_acc,acc_list = evaluate(model, test_dataloader, criterion, epoch, args)

        print("average_acc: ",average_acc,"best_acc1: ",best_acc1)

        is_best = average_acc > best_acc1
        best_acc1 = max(average_acc, best_acc1)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
        )


if __name__ == '__main__':
    same_seeds(42)

    args = arg_parse.args_parse()
    mian(args)
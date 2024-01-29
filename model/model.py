# -*- coding: utf-8 -*-
# @Author  : jacob xu
# @Time    : 2023/7/13 21:33
# @File    : model.py
# @Software: PyCharm


from sklearn.model_selection import KFold
import numpy as np
import torch
from torch.utils.data import random_split,Subset,DataLoader
from gout_data import GoutDataset
import random, math
import torch.nn as nn
from backbone.mlp import mlp
from sklearn.metrics import f1_score, roc_auc_score,precision_score, accuracy_score
from utils import arg_parse
import logging
import shutil

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


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")

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

        feature_embedding = model(batch_features)

        # prototype_list = []
        prototype = []
        for column in range(len(batch_labels[0])):
            column_values = batch_labels[:, column]

            # 根据当前列的值为 1 和 0 的索引，从 train_x 中提取对应的数据
            train_positive = feature_embedding[column_values == 1]
            train_negative = feature_embedding[column_values == 0]

            # 这里相当于得到了正负两类原型
            train_positive_mean = torch.mean(train_positive, dim=0)
            train_negative_mean = torch.mean(train_negative, dim=0)

            # 将原型保存起来
            prototype.append([train_positive_mean, train_negative_mean])

        # 在验证集上进行模型评估
        for val_features, val_labels in val_dataloader:

            val_features = val_features.to(device)
            val_labels = val_labels.to(device)
            predicted_logits_result = []
            predicted_val_result = []

            val_embed = model(val_features)

            for embed in val_embed:
                predicted_class_list = []
                ce_loss = []
                # 对每一个原型都进行计算
                for item in prototype:
                    # 计算每个类别的距离
                    # 这里的距离计算是将每个样本的embedding与原型的embedding进行距离计算
                    # 计算欧式距离
                    pos_euclidean_distance = torch.sqrt(pow(embed - item[0], 2).sum())
                    neg_euclidean_distance = torch.sqrt(pow(embed - item[1], 2).sum())

                    # 计算softmax概率
                    pos_softmax_probs = -torch.exp(pos_euclidean_distance) / (
                            - torch.exp(pos_euclidean_distance) - torch.exp(neg_euclidean_distance))
                    neg_softmax_probs = -torch.exp(neg_euclidean_distance) / (
                            - torch.exp(pos_euclidean_distance) - torch.exp(neg_euclidean_distance))
                    softmax_probs = torch.stack([pos_softmax_probs, neg_softmax_probs])

                    ce_loss.append(softmax_probs)
                    # # 找到具有最高概率的类别索引
                    # predicted_class = torch.argmax(softmax_probs)
                    # predicted_class_list.append(predicted_class)

                predicted_logits = torch.stack(ce_loss, dim=0)
                predicted_logits_result.append(predicted_logits)
                # predicted_result = torch.stack(predicted_class_list, dim=0)
                # predicted_val_result.append(predicted_result)

            # 欧式距离计算得到的分类
            # predicted_val_tensor = torch.stack(predicted_val_result, dim=0)

            predicted_logits_tensor = torch.stack(predicted_logits_result, dim=0)

            positive_col = ["ckd", "heart_failure", "hypertension", "myocardial_infaraction", "nephrolithiasis",
                            "obesity", "stroke", "t2DM", "gout", ]

            ce_loss_list = []
            for column in range(len(val_labels[0])):
                # print("dataloader num: ", i, "disease name:", positive_col[column])

                # task_model = copy.deepcopy(model)
                # task_model.zero_grad()

                val_values = val_labels[:, column]
                # predicted_values = predicted_val_tensor[:, column]
                predicted_logits_values = predicted_logits_tensor[:, column]
                # see = torch.stack([val_values, predicted_values], dim=1).detach().cpu().numpy()

                # mse_loss = nn.MSELoss()
                # mse_value = mse_loss(predicted_values, val_values)


                ce_loss_value = criterion(predicted_logits_values, val_values.long())

                # print("disease name:", positive_col[column], 'CrossEntropyLoss:', ce_loss_value)

                # true_labels = val_values.cpu().numpy()  # 真实标签
                # predicted_labels = predicted_values.cpu().numpy()  # 模型预测结果

                # 计算F1分数
                # f1 = f1_score(true_labels, predicted_labels)

                # 计算AUC
                # auc = roc_auc_score(true_labels, predicted_labels)

                # accuracy = accuracy_score(true_labels, predicted_labels)
                # precision = precision_score(true_labels, predicted_labels)

                # log_metrics(epoch, i, positive_col[column], mse_value, ce_loss_value, f1, auc, accuracy, precision)

                # print("dataloader num: ",i,"disease name:",positive_col[column],'mse_value:', mse_value,
                #       "CrossEntropyLoss:",ce_loss_value, 'f1:', f1, 'auc:', auc, 'accuracy:',accuracy, "precision:",precision)

                ce_loss_list.append(ce_loss_value)

            # 梯度清零
            optimizer.zero_grad()

            # 计算损失的平均值或加权平均值

            # for loss_value in ce_loss_list:
            #     loss_value.backward(retain_graph=True)
            avg_loss = torch.mean(torch.stack(ce_loss_list))
            print("avg_loss:", avg_loss)
            print("++++++++++"*10)
            # 反向传播
            avg_loss.backward()

            # for parameter in model.parameters():
            #     if parameter.requires_grad:
            #         print(f"Parameter {parameter} has been updated: {parameter.grad is not None}")


            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(name, param.grad)

            # 更新模型参数
            optimizer.step()


def evaluate(model, test_dataloader, criterion, epoch, args):

    model.eval()
    with torch.no_grad():

        for i, (batch_features, batch_labels) in enumerate(test_dataloader):

            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            prototype = []

            feature_embedding = model(batch_features)

            for column in range(len(batch_labels[0])):
                column_values = batch_labels[:, column]

                # 根据当前列的值为 1 和 0 的索引，从 train_x 中提取对应的数据
                train_positive = feature_embedding[column_values == 1]
                train_negative = feature_embedding[column_values == 0]

                # 这里是one shot 的思想，从每个类别中随机抽取一个样本作为原型
                random_positive = train_positive[torch.randperm(len(train_positive))[:1]]
                random_negative = train_negative[torch.randperm(len(train_negative))[:1]]

                # 这里相当于得到了正负两类原型
                train_positive_mean = torch.mean(random_positive, dim=0)
                train_negative_mean = torch.mean(random_negative, dim=0)

                # 将原型保存起来
                prototype.append([train_positive_mean, train_negative_mean])

            for test_features, test_labels in test_dataloader:

                test_features = test_features.to(device)
                test_labels = test_labels.to(device)
                predicted_logits_result = []
                predicted_val_result = []

                val_embed = model(test_features)

                for embed in val_embed:
                    predicted_class_list = []
                    ce_loss = []
                    # 对每一个原型都进行计算
                    for item in prototype:
                        # 计算每个类别的距离
                        # 这里的距离计算是将每个样本的embedding与原型的embedding进行距离计算
                        # 计算欧式距离
                        pos_euclidean_distance = torch.sqrt(pow(embed - item[0], 2).sum())
                        neg_euclidean_distance = torch.sqrt(pow(embed - item[1], 2).sum())

                        # 计算softmax概率
                        pos_softmax_probs = -torch.exp(pos_euclidean_distance) / (
                                - torch.exp(pos_euclidean_distance) - torch.exp(neg_euclidean_distance))
                        neg_softmax_probs = -torch.exp(neg_euclidean_distance) / (
                                - torch.exp(pos_euclidean_distance) - torch.exp(neg_euclidean_distance))
                        softmax_probs = torch.stack([pos_softmax_probs, neg_softmax_probs])

                        ce_loss.append(softmax_probs)
                        # # 找到具有最高概率的类别索引
                        predicted_class = torch.argmax(softmax_probs)
                        predicted_class_list.append(predicted_class)

                    predicted_logits = torch.stack(ce_loss, dim=0)
                    predicted_logits_result.append(predicted_logits)
                    predicted_result = torch.stack(predicted_class_list, dim=0)
                    predicted_val_result.append(predicted_result)

                # 欧式距离计算得到的分类
                predicted_val_tensor = torch.stack(predicted_val_result, dim=0)

                predicted_logits_tensor = torch.stack(predicted_logits_result, dim=0)

                positive_col = ["ckd", "heart_failure", "hypertension", "myocardial_infaraction", "nephrolithiasis",
                                "obesity", "stroke", "t2DM", "gout", ]

                ce_loss_list = []
                acc_list = []
                for column in range(len(test_labels[0])):
                    # print("dataloader num: ",i,"disease name:",positive_col[column])

                    test_values = test_labels[:, column]
                    predicted_values = predicted_val_tensor[:, column]
                    predicted_logits_values = predicted_logits_tensor[:, column]
                    see = torch.stack([test_values, predicted_values], dim=1).detach().cpu().numpy()

                    mse_loss = nn.MSELoss()
                    mse_value = mse_loss(predicted_values, test_values)

                    ce_loss_value = criterion(predicted_logits_values, test_values.long())

                    true_labels = test_values.cpu().numpy()  # 真实标签
                    predicted_labels = predicted_values.cpu().numpy()  # 模型预测结果

                    # 计算F1分数
                    f1 = f1_score(true_labels, predicted_labels)

                    # 计算AUC
                    auc = roc_auc_score(true_labels, predicted_labels)

                    accuracy = accuracy_score(true_labels, predicted_labels)
                    precision = precision_score(true_labels, predicted_labels)

                    # log_metrics(i, positive_col[column], mse_value, ce_loss_value, f1, auc, accuracy, precision)

                    # print('mse_value:', mse_value,"CrossEntropyLoss:",ce_loss_value, 'f1:', f1, 'auc:', auc, 'accuracy:',accuracy, "precision:",precision)
                    ce_loss_list.append(ce_loss_value)
                    acc_list.append(accuracy)

                avg_loss = torch.mean(torch.stack(ce_loss_list))
                # print("avg_loss:", avg_loss)

                acc_average = sum(acc_list) / len(acc_list)

                return acc_average,acc_list


def mian(args):
    # 定义k-fold参数和epoch参数
    k = 5  # k-fold的折数
    num_epochs = 10000  # 训练的轮数

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

    for epoch in range(num_epochs):

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
        average_acc, acc_list = evaluate(model, test_dataloader, criterion, epoch, args)

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

        print("average_acc:",average_acc)

        print(acc_list)

    # 在测试集上进行模型评估


if __name__ == '__main__':
    same_seeds(42)

    args = arg_parse.args_parse()
    mian(args)
# -*- coding: utf-8 -*-
# @Author  : jacob xu
# @Time    : 2023/8/1 16:27
# @File    : run_model.py
# @Software: PyCharm
import os

from sklearn.model_selection import KFold
import numpy as np
import torch
from torch.utils.data import random_split,Subset,DataLoader
from dataset import GoutDataset
import random, math
import torch.nn as nn
from backbone import mlp
from utils import arg_parse
import logging
import shutil
import csv
from tqdm import tqdm
from train import train
from train import evaluate
import datetime







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


def save_checkpoint(state, is_best, filename):
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



def run_model(args,train_filename,eval_filename,file_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 定义k-fold参数和epoch参数
    k = 5  # k-fold的折数
    num_epochs = 100  # 训练的轮数

    best_acc1 = 0

    # 创建数据集
    dataset = GoutDataset(
        "/home/xutianxin/Pycharm_code/multi-labels-prototypical-networks/dataset/gout_resampled_data_2.csv")

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
    # csv_file_path = "evaluation_metrics_3.csv"
    with open(file_path+eval_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["disease","MSE","f1_score", "auc", "accuracy", "precision"])  # Header row
        file.close()

    positive_col = ["ckd", "heart_failure", "hypertension", "myocardial_infaraction", "nephrolithiasis",
                    "obesity", "stroke", "t2DM", "gout", ]

    with open(file_path+train_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(positive_col + ["avg_loss"])  # Header row
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

            train(model, train_dataloader, val_dataloader, optimizer, criterion, epoch, args,device,filename=file_path+train_filename)

            print("Fold:", fold + 1)
            print("Epoch:", epoch + 1)
            print("Train samples:", len(train_subset))
            print("Validation samples:", len(val_subset))
            print("------------------------------")

        # 在测试集上进行评估
        test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        average_acc, acc_list = evaluate(model, test_dataloader, criterion, epoch, args,device,filename=file_path+eval_filename)

        print("average_acc: ", average_acc, "best_acc1: ", best_acc1)

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
            filename=file_path+"checkpoint.pth.tar"
        )




if __name__ == '__main__':


    same_seeds(42)
    args = arg_parse.args_parse()


    current_time = datetime.datetime.now().strftime("%H:%M:%S")

    eval_fname = f'/evaluation_metrics_time.csv'
    train_fname = f'/train_loss_values_time.csv'

    file_path =  f'result/protoMaml/{current_time}'

    # Create the parent directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Now you can create the directory
    os.mkdir(file_path)

    logging.basicConfig(filename=file_path + "/training.log",
                        format='%(asctime)s - %(message)s',
                        level=logging.INFO)


    run_model(args,train_fname,eval_fname,file_path)



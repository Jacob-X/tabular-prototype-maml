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
from raw_data import protonet_raw_data
from train import train_balance
from train import evaluate_balance

def log_metrics(epoch,num,name, mse_value, ce_loss_value, f1, auc, accuracy, precision):
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
    num_epochs = 500  # 训练的轮数
    best_acc1 = 0

    train_dataset, test_dataloader, query_dataloader = protonet_raw_data()

    model = mlp(in_features=len(train_dataset.features[0]), out_features=train_dataset.features.shape[1], hidden_sizes=train_dataset.features.shape[1]*2, drop_p=0.5).to(device)

    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    acc_list = []
    with open(file_path+eval_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["disease","MSE","f1_score", "auc", "accuracy", "precision"])  # Header row
        file.close()

    # positive_col = ["ckd", "heart_failure", "hypertension", "myocardial_infaraction", "nephrolithiasis",
    #                 "obesity", "stroke", "t2DM", "gout"]

    positive_col = ["gout", "ckd", "heart_failure", "hypertension", "myocardial_infaraction",
                    "nephrolithiasis", "obesity", "stroke", "t2DM"]

    with open(file_path+train_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(positive_col + ["avg_loss"])  # Header row
        file.close()

    # 数据集是train_dataset, test_dataloader, query_dataloader，不需要进行k_fold交叉验证
    for epoch in tqdm(range(num_epochs)):

        adjust_learning_rate(optimizer, epoch, args)
        train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
        train_balance(model, train_dataloader, query_dataloader, optimizer, criterion, epoch, args, device,
              filename=file_path + train_filename)

        print("Epoch:", epoch + 1)

        average_acc, acc_list = evaluate_balance(model, test_dataloader, criterion, epoch, args,device,filename=file_path+eval_filename)

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

    same_seeds(444)
    args = arg_parse.args_parse()

    current_time = datetime.datetime.now().strftime("%H:%M:%S")

    eval_fname = f'/evaluation_metrics.csv'
    train_fname = f'/train_loss_values.csv'

    file_path = f'result/protoMaml/{current_time}'

    # Create the parent directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Now you can create the directory
    os.mkdir(file_path)

    logging.basicConfig(filename=file_path + "/training.log",
                        format='%(asctime)s - %(message)s',
                        level=logging.INFO)

    run_model(args,train_fname,eval_fname,file_path)



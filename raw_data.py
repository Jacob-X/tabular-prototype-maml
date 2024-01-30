# -*- coding: utf-8 -*-
# @Author  : jacob xu
# @Time    : 2023/8/22 21:30
# @File    : raw_data.py
# @Software: PyCharm

from torch.utils.data import DataLoader
from dataset import goutDataset
from dataset import comorbidityDataset
import torch

def protonet_raw_data():
    # 新的数据集
    train_dataset = goutDataset("/home/xutianxin/Pycharm_code/self_supervised_protonet/comorbidity_data/self_supervised_data.csv")

    comorbidities_path = "/home/xutianxin/Pycharm_code/self_supervised_protonet/comorbidity_data/"

    single_disease_path = "/home/xutianxin/Pycharm_code/self_supervised_protonet/comorbidity_data/"

    gout_heart_failure_dataset = comorbidityDataset(comorbidities_path + "gout_heart_failure.csv")
    gout_myocardial_infaraction_dataset = comorbidityDataset(comorbidities_path + "gout_myocardial_infarction.csv")
    gout_hypertension_dataset = comorbidityDataset(comorbidities_path + "gout_hypertension.csv")
    gout_t2DM_dataset = comorbidityDataset(comorbidities_path + "gout_t2DM.csv")
    gout_stroke_dataset = comorbidityDataset(comorbidities_path + "gout_stroke.csv")
    gout_nephrolithiasis_dataset = comorbidityDataset(comorbidities_path+"gout_nephrolithiasis.csv")
    gout_obesity_dataset = comorbidityDataset(comorbidities_path+"gout_obesity.csv")
    gout_ckd_dataset = comorbidityDataset(comorbidities_path+"gout_ckd.csv")

    gout_heart_failure = DataLoader(gout_heart_failure_dataset, batch_size=len(gout_heart_failure_dataset),shuffle=False)
    gout_myocardial_infaraction = DataLoader(gout_myocardial_infaraction_dataset, batch_size=len(gout_myocardial_infaraction_dataset), shuffle=False)
    gout_hypertension = DataLoader(gout_hypertension_dataset, batch_size=len(gout_hypertension_dataset), shuffle=False)
    gout_t2DM = DataLoader(gout_t2DM_dataset, batch_size=len(gout_t2DM_dataset), shuffle=False)
    gout_stroke = DataLoader(gout_stroke_dataset, batch_size=len(gout_stroke_dataset), shuffle=False)
    gout_nephrolithiasis = DataLoader(gout_nephrolithiasis_dataset, batch_size=len(gout_nephrolithiasis_dataset), shuffle=False)
    gout_obesity = DataLoader(gout_obesity_dataset, batch_size=len(gout_obesity_dataset), shuffle=False)
    gout_ckd = DataLoader(gout_ckd_dataset, batch_size=len(gout_ckd_dataset), shuffle=False)

    # 创建测试集,这个是没有顺序要求的
    test_dataloader = []
    test_dataloader.append(gout_heart_failure)
    test_dataloader.append(gout_myocardial_infaraction)
    test_dataloader.append(gout_hypertension)
    test_dataloader.append(gout_t2DM)
    test_dataloader.append(gout_stroke)
    test_dataloader.append(gout_nephrolithiasis)
    test_dataloader.append(gout_obesity)
    test_dataloader.append(gout_ckd)

    # 创建单病种测试集
    heart_failure_dataset = comorbidityDataset(comorbidities_path + "heart_failure.csv")
    myocardial_infaraction_dataset = comorbidityDataset(comorbidities_path + "myocardial_infarction.csv")
    hypertension_dataset = comorbidityDataset(comorbidities_path + "hypertension.csv")
    t2DM_dataset = comorbidityDataset(comorbidities_path + "t2d.csv")
    stroke_dataset = comorbidityDataset(comorbidities_path + "stroke.csv")
    nephrolithiasis_dataset = comorbidityDataset(comorbidities_path+"Nephrolithiasis.csv")
    obesity_dataset = comorbidityDataset(comorbidities_path+"obesity.csv")
    ckd_dataset = comorbidityDataset(comorbidities_path+"ckd.csv")
    gout_dataset = comorbidityDataset(comorbidities_path+"gout.csv")

    heart_failure = DataLoader(heart_failure_dataset, batch_size=len(heart_failure_dataset),shuffle=False)
    myocardial_infaraction = DataLoader(myocardial_infaraction_dataset, batch_size=len(myocardial_infaraction_dataset), shuffle=False)
    hypertension = DataLoader(hypertension_dataset, batch_size=len(hypertension_dataset), shuffle=False)
    t2DM = DataLoader(t2DM_dataset, batch_size=len(t2DM_dataset), shuffle=False)
    stroke = DataLoader(stroke_dataset, batch_size=len(stroke_dataset), shuffle=False)
    nephrolithiasis = DataLoader(nephrolithiasis_dataset, batch_size=len(nephrolithiasis_dataset), shuffle=False)
    obesity = DataLoader(obesity_dataset, batch_size=len(obesity_dataset), shuffle=False)
    ckd = DataLoader(ckd_dataset, batch_size=len(ckd_dataset), shuffle=False)
    gout = DataLoader(gout_dataset, batch_size=len(gout_dataset), shuffle=False)

    # 要和训练集保持一样的顺序
    query_dataloader = []
    query_dataloader.append(gout)
    query_dataloader.append(ckd)
    query_dataloader.append(heart_failure)
    query_dataloader.append(hypertension)
    query_dataloader.append(myocardial_infaraction)
    query_dataloader.append(nephrolithiasis)
    query_dataloader.append(obesity)
    query_dataloader.append(stroke)
    query_dataloader.append(t2DM)

    return train_dataset,test_dataloader,query_dataloader

if __name__ == '__main__':

    protonet_raw_data()
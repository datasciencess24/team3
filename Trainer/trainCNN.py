import torch
import os
import argparse
from torch.nn.utils.rnn import pad_sequence
import json
import logging
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import heapq
import numpy as np
import torch

from Models.CNN import TimeSeriesCNN



def train_timeseries_Conv(train_dataset, device, args, sequence_length):  # 输入了节点的标签和样本的原始字节，有好多好多列 #把这个函数在trainer里边改
    # Hyper parameters
    loss_list = []
    # # size = 600
    # size = len(sample_raw_bytes[0])  # 取有多少个样本
    # num_epochs = 100  # 训练数量 100
    # batch_size = 100 # 每批训练的大小 100
    # learning_rate = 0.001  # 学习率
    # train_features_tensor = torch.from_numpy(sample_raw_bytes)  # 把数组转换成张量
    # train_labels_tensor = torch.from_numpy(labels).long()
    batch_size = args.train_batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    #size = len(train_dataset[0]) # number of sample
    # train_feature_shape = train_features_tensor.shape  # 获得训练的特征的形状（row,colum） 相当于就是有几行几列
    # x = torch.ones((train_feature_shape[0], 1, train_feature_shape[1]))  # 创建一个三维数组，然后里边的元素全部都用1填充（raw_bytes,1,labels）
    # for i in range(train_feature_shape[0]):
    #     x[i][0][:] = train_features_tensor[i][:]
    # features_tensor = x.long()
    #
    # train_dataset = torch.utils.data.TensorDataset(train_features_tensor, train_labels_tensor)  # 把原始字节和label放到一个新的数据集里

    #train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)  # 然后再分批加载出来
    model = TimeSeriesCNN(args, device, sequence_length).to(device)  # 三个参数： number_of_feature, deopout,device
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 用于寻找模型的最优解

    # Train the model
    total_step = len(train_dataset)  # train dataset的总的步长，即有多少要训练的数据
    for epoch in range(num_epochs):  # 要训练几次模型
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_dataset):  # train_loader中含有 feature 和 label
            if (i + 1) % 2 == 0:  # 99，199，299，399...
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)  # 将feature放入之前建立好的模型中
                predicted = outputs.max(dim=1)[1]
                total += labels.size(0)  # 总训练数
                correct += (predicted == labels).sum().item()  # 计算正确率

                print('Valid Accuracy of the model on the valid dataset: {} %'.format(100 * correct / total))
            else:
                images,labels = images.to(device),labels.to(device)
                labels = labels.long()
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)  # 计算损失，损失越小说明预测越准确

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_value = loss.item()
            loss_list.append(loss_value)
    # sava the model
    model_path = '../cache/'
    if (os.path.exists(model_path) == False):
        os.makedirs(model_path)
    torch.save(model, model_path + 'Timeseries_Conv_model_1.pkl')
    return model

# -*- coding: utf-8 -*-
# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For data preprocess
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# 根据打印出的文件路径，将文件读取进内存
sample_submission = pd.read_csv('sample_submission.csv')
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 我们先看一下这三个数据长什么样
sample_submission.shape, train_data.shape, test_data.shape

"""从数字上我们可以看出以下两点：
1. 我们要提交的文件只包含两列，第一列是对应了每一个测试集样本的`Id`，第二列是我们所预测的`SalePrice`(如上课所演示的那样)
1. 上课演示的训练集与测试集的形状分别为`(1460, 81)`和`(1459, 80)`，可以看出比赛用的数据集样本数量更多，但是特征却更少了
"""

# 我们先来粗略看一下数据集包含哪些特征，head方法可以查看前五行数据
train_data.head()

"""可以发现`Id`为4的数据缺失了许多特征，像这样的数据应该还会有很多

但是不用慌，老师在上课的时候已经教会我们要如何处理这些缺失的数据

不记得的话可以回去b站看回放哦~
"""

# 保险起见，还是看一下测试集长什么样子再进行下一步处理
test_data.head()

"""注意到比赛用到的训练集标签(即真实成交价格)与上课时用到的不同，上课时用到的标签在最后一列，而比赛用到的标签在第三列，名称为`Sold Price`

通过搜索，我在[stackoverflow](https://stackoverflow.com/questions/29763620/how-to-select-all-columns-except-one-column-in-pandas)上查到如何去掉某列，那现在我们继续把数据集的特征提取出来吧~
"""

# train_data.loc[:, train_data.columns != 'Sold Price'] # 这行代码用于提取除'Sold Price'外的其他列
all_features = pd.concat((train_data.loc[:, train_data.columns != 'Sold Price'], test_data.iloc[:, 1:]))
all_features.info() # info方法可以总览数据

# 将所有缺失的值替换为相应特征的平均值。通过将特征重新缩放到零均值和单位方差来标准化数据
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# # 处理离散值。我们用一次独热编码替换它们
# all_features = pd.get_dummies(all_features, dummy_na=True)
# all_features.shape

"""如果我们直接用老师提供的代码(即上面被注释掉的代码)，那kaggle就会报一个错误：

> Your notebook tried to allocate more memory than is available. It has restarted.

这说明特征分布太过分散了，如果用独热编码就会占用大量内存，导致重启

简单起见，接下来我尝试跳过对字符串的处理，只利用数值特征进行训练
"""

all_features = all_features[numeric_features[1:]] # 原本第一列是Id，去掉
all_features.info()

"""可以看出，现在的特征只包含18列了，并且都是数值特征"""

import torch

# 从pandas格式中提取NumPy格式，并将其转换为张量表示
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values,
                              dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values,
                             dtype=torch.float32)
# 注意课上数据的标签列为SalePrice，与比赛用的标签列名不同
train_labels = torch.tensor(train_data['Sold Price'].values.reshape(-1, 1),
                            dtype=torch.float32)

from torch import nn

# 定义模型与损失函数
loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(
        nn.Linear(in_features, 64),
        nn.Linear(64, 1),
        )

    return net

# 我们更关心相对误差，解决这个问题的一种方法是用价格预测的对数来衡量差异
def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()

from torch.utils import data

# Defined in file: ./chapter_linear-networks/linear-regression-concise.md
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器。"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

# 我们的训练函数将借助Adam优化器
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    # 注意kaggle环境中没有安装d2l，故用到的地方需要手动定义
    train_iter = load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

# K折交叉验证
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

# 返回训练和验证误差的平均值
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        # 删去利用d2l画图的代码
        print(f'fold {i + 1}, train log rmse {float(train_ls[-1]):f}, '
              f'valid log rmse {float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

# 模型选择
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.1, 0.001, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')

"""考虑到训练集有47439条数据，做交叉验证的时间会比较长，大约需要5分钟

此处超参数直接照搬课件，故损失值比较大
"""

# 最后提交前需要确认比赛所需的格式是否和课上数据集有出入
sample_submission.head()

# 模型选择
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.1, 0.001, 64
# 提交你的Kaggle预测
def train_and_pred(train_features, test_feature, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    # 删去利用d2l画图的代码
    print(f'train log rmse {float(train_ls[-1]):f}')
    preds = net(test_features).detach().numpy()
    # 不出所料，列名需要做替换
    test_data['Sold Price'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['Sold Price']], axis=1)
    submission.to_csv('submission.csv', index=False)
    # 最后返回一下提交的结果，以便查看
    return submission

submission = train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)

# 查看一下预测结果，以确保格式与样例一致
submission.head()

"""至此就可以向kaggle提交我们的结果了~

如果这篇笔记本对你有帮助，不妨点个赞以满足我小小的虚荣心，爱你哟~
"""
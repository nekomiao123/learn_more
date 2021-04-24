# -*- coding: utf-8 -*-
# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For data preprocess
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import csv
import math
# use this to record my loss
import wandb


# Specify the graphics card
torch.cuda.set_device(3)

train_path = 'train.csv'                # path to training data
test_path = 'test.csv'                  # path to testing data

def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()

train_name = 'warmup_and_feature41'
config = {
    'n_epochs': 1000,                # maximum number of epochs
    'batch_size': 32,                # mini-batch size for dataloader
    'optimizer': 'Adam',             # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.001,                 # learning rate of Adam
        'weight_decay': 0.001        # weight decay 
    },
    'early_stop': 500,                  # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': train_name + '_model.pth',        # your model will be saved here
    'warm_up_epochs': 5
}

default_config = dict(
    batch_size=config['batch_size'],
    n_epochs=config['n_epochs'],
    optimizer=config['optimizer'],
    optim_hparas=config['optim_hparas'],
    early_stop=config['early_stop'],
    warm_up_epochs=config['warm_up_epochs']
)

# 初始化该run
wandb.init(project='chp', entity='nekokiku', config=default_config, name=train_name)

class HousePriceDataset(Dataset):
    def __init__(self, path, mode='train', select_features=True):
        self.mode = mode

        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        # train_data.loc[:, train_data.columns != 'Sold Price'] # 这行代码用于提取除'Sold Price'外的其他列
        all_features = pd.concat((train_data.loc[:, train_data.columns != 'Sold Price'], test_data.iloc[:, 1:]))

        # 将所有缺失的值替换为相应特征的平均值。通过将特征重新缩放到零均值和单位方差来标准化数据
        numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
        all_features[numeric_features] = all_features[numeric_features].apply(
            lambda x: (x - x.mean()) / (x.std()))
        all_features[numeric_features] = all_features[numeric_features].fillna(0)

        # # 处理离散值。我们用一次独热编码替换它们,占用的内存很大
        # all_features = pd.get_dummies(all_features, dummy_na=True)
        # all_features.shape
        # 只利用数值特征进行训练
        all_features = all_features[numeric_features[1:]] # 原本第一列是Id，去掉

        if select_features:
        # high correlation(Annual tax amount 15, Tax assessed value 14, Last Sold Price 17, Listed Price 16, Full bathrooms 4, Bathrooms 3)
            feats = [14, 15, 16, 17, 4, 3]
        else:
            feats = list(range(all_features.shape[1]))

        # 从pandas格式中提取NumPy格式，并将其转换为张量表示
        n_train = train_data.shape[0]
        new_features = all_features[:].values
        if mode == 'test':
            self.data = torch.FloatTensor(new_features[n_train:,feats])

        else:
            data = new_features[:n_train,feats]
            labels = train_data['Sold Price'].values.reshape(-1, 1)

            # Splitting training data into train & dev sets 4:1
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 5 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 5 == 0]

            # Convert data into PyTorch tensors
            self.data = torch.FloatTensor(data[indices])
            self.labels = torch.FloatTensor(labels[indices])

        self.dim = self.data.shape[1]
        print('Finished reading the {} set of HousePriceDataset Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))


    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.labels[index]
        else:
            # For testing (no target)
            return self.data[index]

    def __len__(self):
        return len(self.data)


def prep_dataloader(path, mode, batch_size, n_jobs=2):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = HousePriceDataset(path, mode=mode)  # Construct dataset
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)                            # Construct dataloader
    return dataloader


class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        # Define your neural network here
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        return self.net(x)

    def cal_loss(self, pred, target):   
        ''' Calculate loss rmse'''
        eps = 1e-6
        loss = self.criterion(pred, target)
        newloss = torch.sqrt(loss + eps)
        return newloss

def log_rmse(preds, labels):
    loss = nn.MSELoss()
    clipped_preds = torch.clamp(preds, 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()

def train(train_loader, dev_loader, config, device):

    ''' DNN training '''
    in_features = train_loader.dataset.dim
    # define a loss function, and optimizer
    model = NeuralNet(in_features).to(device)
    # Maximum number of epochs
    n_epochs = config['n_epochs']  
    # Setup optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])

    # learning rate schedule
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    # warm_up_with_cosine_lr
    warm_up_with_cosine_lr = lambda epoch: epoch / config['warm_up_epochs'] if epoch <= config['warm_up_epochs'] else 0.5 * ( math.cos((epoch - config['warm_up_epochs']) /(n_epochs - config['warm_up_epochs']) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR( optimizer, lr_lambda=warm_up_with_cosine_lr)

    wandb.watch(model)
    min_mse = 1000.
    early_stop_cnt = 0
    epoch = 0

    while epoch < n_epochs:
        train_loss = []
        dev_loss = []
        model.train()
        train_total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad() 
            x, y = x.to(device), y.to(device)
            pred = model(x)
            log_rmse_loss = model.cal_loss(pred, y)
            log_rmse_loss.backward()
            optimizer.step()

            train_loss.append(log_rmse(pred, y))

        train_total_loss = sum(train_loss) / len(train_loss)
        # print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_total_loss:.5f}")

        # After each epoch, test your model on the validation (development) set.
        model.eval() 
        val_total_loss = 0
        for x, y in dev_loader:
            x, y = x.to(device), y.to(device) 
            with torch.no_grad():
                pred = model(x) 
                val_loss = model.cal_loss(pred, y)
            dev_loss.append(log_rmse(pred, y))

        val_total_loss = sum(dev_loss) / len(dev_loss)

        print(f"[ {epoch + 1:03d}/{n_epochs:03d} ] train_loss = {train_total_loss:.5f} val_loss = {val_total_loss:.5f}")

        # learning rate decay and print 
        # warm up
        scheduler.step()
        realLearningRate = scheduler.get_last_lr()[0]

        # ReduceLROnPlateau
        # scheduler.step(val_total_loss)
        # realLearningRate = optimizer.state_dict()['param_groups'][0]['lr']

        # wandb
        wandb.log({'epoch': epoch + 1, 'train_loss': train_total_loss, 'val_loss': val_total_loss, 'learningRate':realLearningRate})

        if val_total_loss < min_mse:
            # Save model if your model improved
            min_mse = val_total_loss
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                .format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        dev_loss.append(val_total_loss)
        if early_stop_cnt > config['early_stop']:
            # Stop training if your model stops improving for "config['early_stop']" epochs.
            break

    print('Finished training after {} epochs'.format(epoch))

def test_save(test_loader, file, device, test_data):

    in_features = test_loader.dataset.dim
    # create model and load weights from checkpoint
    model = NeuralNet(in_features).to(device)
    model.load_state_dict(torch.load(config['save_path']))

    model.eval()                                # set model to evalutation mode
    preds = []
    for x in test_loader:                            # iterate through the dataloader
        x = x.to(device)                        # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            preds.append(pred.detach().cpu())   # collect prediction
    preds = torch.cat(preds, dim=0).numpy()     # concatenate all predictions and convert to a numpy array

    #  ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    test_data['Sold Price'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['Sold Price']], axis=1)
    submission.to_csv(file, index=False)

def main():
    batch_size = config['batch_size']
    device = get_device()
    print('begin to load training data')
    train_loader = prep_dataloader(train_path, 'train', batch_size)
    dev_loader = prep_dataloader(train_path, 'dev', batch_size)
    print('loading trainning data complete')

    print('begin to train')
    train(train_loader, dev_loader, config, device)
    print('trainning complete')

    print("testing")
    test_data = pd.read_csv(test_path)
    test_loader = prep_dataloader(test_path, 'test', batch_size)
    test_save(test_loader, train_name + 'pred.csv', device, test_data)
    print("testing complete")


if __name__ == '__main__':
    main()

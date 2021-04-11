# -*- coding: utf-8 -*-
'''
The DARPA TIMIT Acoustic-Phonetic Continuous Speech Corpus (TIMIT)
The TIMIT corpus of reading speech has been designed to provide speech data for the acquisition of acoustic-phonetic knowledge and for the development and evaluation of automatic speech recognition systems.

This homework is a multiclass classification task, 
we are going to train a deep neural network classifier to predict the phonemes for each frame from the speech corpus TIMIT.

link: https://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3

You should have `timit_11/train_11.npy`, `timit_11/train_label_11.npy`, and `timit_11/test_11.npy` after running this block.<br><br>
`timit_11/`
- `train_11.npy`: training data
- `train_label_11.npy`: training label
- `test_11.npy`:  testing data

This is the code for ML2021's homework by HUNG-YI LEE 
Homework 2-1 Phoneme Classification
I just rewrite the code to practice my skill.

'''

# import different library
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import gc
import torch
import torch.nn as nn
# Specify the graphics card
torch.cuda.set_device(6)
# log file
from loguru import logger
logger.add('train.log')
# tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# hyper-parameters 
config = {
    'learning_rate': 0.0001,       # learning rate
    'batch_size': 256,
    'num_epoch': 200,
    'val_ratio': 0.2,       
    'model_path': './train_model/model.ckpt'
}

## Create Dataset
class TIMITDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(np.int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

## Create Models
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(429),
            nn.Linear(429, 4096),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(128, 39)
        )

    def forward(self, x):
        return self.net(x)

# load and process the data
def prep_dataloader(batch_size, val_ratio, data_root='./timit_11/'):

    train = np.load(data_root + 'train_11.npy')
    train_label = np.load(data_root + 'train_label_11.npy')
    test = np.load(data_root + 'test_11.npy')

    logger.debug('Size of training data: {}'.format(train.shape))
    logger.debug('Size of testing data: {}'.format(test.shape))
    logger.debug('Size of label data: {}'.format(train_label.shape))

    """Split the labeled data into a training set and a validation set, 
    you can modify the variable `val_ratio` to change the ratio of validation data."""

    percent = int(train.shape[0] * (1 - val_ratio))
    train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
    logger.debug('Size of training set: {}'.format(train_x.shape))
    logger.debug('Size of validation set: {}'.format(val_x.shape))


    """Create a data loader from the dataset, feel free to tweak the variable `batch_size` here."""

    train_set = TIMITDataset(train_x, train_y)
    val_set = TIMITDataset(val_x, val_y)
    train_loader = DataLoader(train_set, batch_size, shuffle=True) #only shuffle the training data
    val_loader = DataLoader(val_set, batch_size, shuffle=False)

    # create testing dataset
    test_set = TIMITDataset(test, None)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, test_set, train_set, val_set

#check device
def get_device():
  return 'cuda' if torch.cuda.is_available() else 'cpu'

# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# train
def train_data(train_loader, val_loader, train_set, val_set, config, device):

    # define a loss function, and optimizer
    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    num_epoch = config['num_epoch']
    model_path = config['model_path']
    # start training

    best_acc = 0.0
    for epoch in range(num_epoch):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        # training
        model.train() # set the model to training mode
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() 
            outputs = model(inputs) 
            batch_loss = criterion(outputs, labels)
            _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
            batch_loss.backward() 
            optimizer.step() 

            train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
            train_loss += batch_loss.item()

        # validation
        if len(val_set) > 0:
            model.eval() # set the model to evaluation mode
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    batch_loss = criterion(outputs, labels) 
                    _, val_pred = torch.max(outputs, 1) 
                
                    val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
                    val_loss += batch_loss.item()


                logger.debug('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                    epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader), val_acc/len(val_set), val_loss/len(val_loader)
                ))
                # use tensorboard to visualize the accuracy and loss
                real_train_acc = train_acc/len(train_set)
                real_train_loss = train_loss/len(train_loader)
                real_val_acc = val_acc/len(val_set)
                real_val_loss = val_loss/len(val_loader)
                writer.add_scalar("Acc/Val", real_val_acc, epoch)
                writer.add_scalar("Loss/Val", real_val_loss, epoch)
                writer.add_scalar("Acc/train", real_train_acc, epoch)
                writer.add_scalar("Loss/train", real_train_loss, epoch)

                # if the model improves, save a checkpoint at this epoch
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), model_path)
                    logger.debug('saving model with acc {:.3f}'.format(best_acc/len(val_set)))
        else:
            logger.debug('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader)
            ))

    writer.flush()
    writer.close()
    # if not validating, save the last epoch
    if len(val_set) == 0:
        torch.save(model.state_dict(), model_path)
        logger.debug('saving model at last epoch')

# predict
def predict_data(test_loader, device):

    # create model and load weights from checkpoint
    model = Classifier().to(device)
    model.load_state_dict(torch.load(config['model_path']))

    predict = []
    model.eval() # set the model to evaluation mode
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability

            for y in test_pred.cpu().numpy():
                predict.append(y)
        
    with open('prediction.csv', 'w') as f:
        f.write('Id,Class\n')
        for i, y in enumerate(predict):
            f.write('{},{}\n'.format(i, y))
        
    logger.debug("prediect done!!!")

def main():
    # get device 
    logger.debug('------------------------------------------------------------------------------------------------------')
    device = get_device()
    logger.debug(f'DEVICE: {device}')
    same_seeds(0)
    batch_size = config['batch_size']
    val_ratio = config['val_ratio']
    logger.debug('New train begins')
    logger.debug('Loading data ...')
    train_loader, val_loader, test_loader, test_set, train_set, val_set = prep_dataloader(batch_size, val_ratio)
    logger.debug('Loading data complete')
    train_data(train_loader, val_loader, train_set, val_set, config, device)
    predict_data(test_loader, device)
    logger.debug('All done')

if __name__ == '__main__':
    main()


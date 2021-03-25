# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1pY8kdqe32ITrmZmqMyKpJGvGJ9OHkSGw

Homework 2-1 Phoneme Classification

The DARPA TIMIT Acoustic-Phonetic Continuous Speech Corpus (TIMIT)
The TIMIT corpus of reading speech has been designed to provide speech data for the acquisition of acoustic-phonetic knowledge and for the development and evaluation of automatic speech recognition systems.

This homework is a multiclass classification task, 
we are going to train a deep neural network classifier to predict the phonemes for each frame from the speech corpus TIMIT.

link: https://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3

You should have `timit_11/train_11.npy`, `timit_11/train_label_11.npy`, and `timit_11/test_11.npy` after running this block.<br><br>
`timit_11/`
- `train_11.npy`: training data<br>
- `train_label_11.npy`: training label<br>
- `test_11.npy`:  testing data<br><br>

notes: if the google drive link is dead, you can download the data directly from Kaggle and upload it to the workspace
"""

# import different library
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import gc
import torch
import torch.nn as nn

# log file
from loguru import logger

logger.add('train.log')

# Specify the graphics card
torch.cuda.set_device(0)

''' hyperparameters '''

# the ratio of validation data
VAL_RATIO = 0.2   

BATCH_SIZE = 64

# training parameters
num_epoch = 30               # number of training epoch
learning_rate = 0.0001       # learning rate

# the path where checkpoint saved
model_path = './train_model/model.ckpt'

""" Preparing Data
Load the training and testing data from the `.npy` file (NumPy array).
"""

logger.debug('------------------------------------------------------------------------------------------------------')
logger.debug('New train begins')
logger.debug('Loading data ...')

data_root='./timit_11/'
train = np.load(data_root + 'train_11.npy')
logger.debug('Loading train data complete')
train_label = np.load(data_root + 'train_label_11.npy')
logger.debug('Loading train label data complete')
test = np.load(data_root + 'test_11.npy')
logger.debug('Loading test data complete')

logger.debug('Size of training data: {}'.format(train.shape))
logger.debug('Size of testing data: {}'.format(test.shape))
logger.debug('Size of label data: {}'.format(train_label.shape))

"""## Create Dataset"""

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


"""Split the labeled data into a training set and a validation set, you can modify the variable `VAL_RATIO` to change the ratio of validation data."""

percent = int(train.shape[0] * (1 - VAL_RATIO))
train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
logger.debug('Size of training set: {}'.format(train_x.shape))
logger.debug('Size of validation set: {}'.format(val_x.shape))


"""Create a data loader from the dataset, feel free to tweak the variable `BATCH_SIZE` here."""

train_set = TIMITDataset(train_x, train_y)
val_set = TIMITDataset(val_x, val_y)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) #only shuffle the training data
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)


"""Cleanup the unneeded variables to save memory.<br>

**notes: if you need to use these variables later, then you may remove this block or clean up unneeded variables later
the data size is quite huge, so be aware of memory usage in colab
"""
del train, train_label, train_x, train_y, val_x, val_y
gc.collect()


"""## Create Model
Define model architecture, you are encouraged to change and experiment with the model architecture.
"""

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(429, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 39)
        )

    def forward(self, x):
        return self.net(x)
        

"""## Training"""

#check device
def get_device():
  return 'cuda' if torch.cuda.is_available() else 'cpu'

# get device 
device = get_device()
logger.debug(f'DEVICE: {device}')

"""Fix random seeds for reproducibility."""

# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# fix random seed for reproducibility
same_seeds(0)

# create model, define a loss function, and optimizer
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                logger.debug('saving model with acc {:.3f}'.format(best_acc/len(val_set)))
    else:
        logger.debug('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader)
        ))

# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    logger.debug('saving model at last epoch')

""" Testing

Create a testing dataset, and load model from the saved checkpoint.
"""

# create testing dataset
test_set = TIMITDataset(test, None)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# create model and load weights from checkpoint
model = Classifier().to(device)
model.load_state_dict(torch.load(model_path))

"""Make prediction."""

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

"""Write prediction to a CSV file.

After finish running this block, download the file `prediction.csv` from the files section on the left-hand side and submit it to Kaggle.
"""

with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predict):
        f.write('{},{}\n'.format(i, y))

logger.debug("done!!!")
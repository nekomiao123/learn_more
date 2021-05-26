# -*- coding: utf-8 -*-

# Import packages
import numpy as np
import random
import torch
import math

from torch.utils.data import DataLoader
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import torchvision.transforms as transforms

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

from torch.optim import Adam, AdamW

from sklearn.cluster import MiniBatchKMeans
from scipy.cluster.vq import vq, kmeans

from qqdm import qqdm, format_str
import pandas as pd
# use pdb.set_trace() to set breakpoints for debugging
import pdb  
# use this to record my loss
import wandb

torch.cuda.set_device(1)
"""# Loading data"""

train_path = 'data-bin/trainingset.npy'
test_path = 'data-bin/testingset.npy'

train_name = '1000_Medium_cnn'

# hyperparameter
default_config = dict(
    batch_size=128,                 # medium: smaller batchsize
    num_epochs=1000,
    learning_rate=3e-3,             # learning rate of Adam
    model_type='cnn',               # selecting a model type from {'cnn', 'fcn', 'vae', 'resnet'}

    early_stop=200,
    warm_up_epochs=5
)

# Initialize 
wandb.init(project='hw08', entity='nekokiku', config=default_config, name=train_name)
config = wandb.config

"""## Random seed
Set the random seed to a certain value for reproducibility.
"""
def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(19530615)

class fcn_autoencoder(nn.Module):
    def __init__(self):
        super(fcn_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64 * 64 * 3, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), 
            nn.Linear(64, 12), 
            nn.ReLU(True), 
            nn.Linear(12, 3))
        
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), 
            nn.Linear(128, 64 * 64 * 3), 
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),  # 32x32x12
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1), # 16x16x24
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1), # 8x8x48
            nn.BatchNorm2d(48),
            nn.ReLU(),
            # nn.Conv2d(48, 96, 4, stride=2, padding=1), # 4x4x96
            # nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1), # 8x8x48
            # nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1), # 16x16x24
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1), # 32x32x12
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),  # 64x64x3
            nn.BatchNorm2d(3),
            nn.Tanh(), 
        )

    def forward(self, x):
        x = self.encoder(x)

        x = self.decoder(x)
        return x 

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),    
            nn.ReLU(),
        )
        self.enc_out_1 = nn.Sequential(
            nn.Conv2d(24, 48, 4, stride=2, padding=1),  
            nn.ReLU(),
        )
        self.enc_out_2 = nn.Sequential(
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1), 
            nn.Tanh(),
        )

    def encode(self, x):
        h1 = self.encoder(x)
        return self.enc_out_1(h1), self.enc_out_2(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_vae(recon_x, x, mu, logvar, criterion):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    mse = criterion(recon_x, x)  # mse loss
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return mse + KLD

class Resnet(nn.Module):
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, drop_p=0.3, CNN_embed_dim=256):
        super(Resnet, self).__init__()

        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # encoding components
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]                            # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)

        self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)      # output = CNN embedding latent variables

        # Sampling vector
        self.fc4 = nn.Linear(self.CNN_embed_dim, self.fc_hidden2)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc5 = nn.Linear(self.fc_hidden2, 64 * 4 * 4)
        self.fc_bn5 = nn.BatchNorm1d(64 * 4 * 4)
        self.relu = nn.ReLU(inplace=True)

        # Decoder
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()    # y = (y1, y2, y3) \in [0 ,1]^3
        )


    def encode(self, x):
        x = self.resnet(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        if x.shape[0] > 1:
            x = self.bn1(self.fc1(x))
        else:
            x = self.fc1(x)
        x = self.relu(x)
        if x.shape[0] > 1:
            x = self.bn2(self.fc2(x))
        else:
            x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3_mu(x)
        return x

    def decode(self, z):
        if z.shape[0] > 1:
            x = self.relu(self.fc_bn4(self.fc4(z)))
            x = self.relu(self.fc_bn5(self.fc5(x))).view(-1, 64, 4, 4)
        else:
            x = self.relu(self.fc4(z))
            x = self.relu(self.fc5(x)).view(-1, 64, 4, 4)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=True)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_reconst = self.decode(z)

        return x_reconst

# Dataset module
# Module for obtaining and processing data. The transform function here normalizes image's pixels from [0, 255] to [-1.0, 1.0].
class CustomTensorDataset(TensorDataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors):
        self.tensors = tensors
        if tensors.shape[-1] == 3:
            self.tensors = tensors.permute(0, 3, 1, 2)

        self.transform = transforms.Compose([
                            transforms.Lambda(lambda x: x.to(torch.float32)),
                            transforms.Lambda(lambda x: 2. * x/255. - 1.),
                            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                            ])

    def __getitem__(self, index):
        x = self.tensors[index]
        
        if self.transform:
            # mapping images to [-1.0, 1.0]
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.tensors)

def prep_dataloader(batch_size):
    train = np.load(train_path, allow_pickle=True)
    test = np.load(test_path, allow_pickle=True)
    # Build training dataloader
    x = torch.from_numpy(train)
    train_dataset = CustomTensorDataset(x)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)


    # build testing dataloader
    data = torch.tensor(test, dtype=torch.float32)
    test_dataset = CustomTensorDataset(data)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size, num_workers=1)

    return train_dataloader, test_dataloader

def train(num_epochs, learning_rate, train_dataloader, model_type):
    # Model
    model_classes = {'resnet': Resnet(), 'fcn':fcn_autoencoder(), 'cnn':conv_autoencoder(), 'vae':VAE(), }
    model = model_classes[model_type].cuda()

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate)

    # Training loop
    best_loss = np.inf
    model.train()

    # learning rate schedule
    warm_up_with_cosine_lr = lambda epoch: epoch / config['warm_up_epochs'] if epoch <= config['warm_up_epochs'] else 0.5 * ( math.cos((epoch - config['warm_up_epochs']) /(num_epochs - config['warm_up_epochs']) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR( optimizer, lr_lambda=warm_up_with_cosine_lr)

    qqdm_train = qqdm(range(num_epochs), desc=format_str('bold', 'Description'))
    for epoch in qqdm_train:
        tot_loss = list()
        for data in train_dataloader:

            # ===================loading=====================
            if model_type in ['cnn', 'vae', 'resnet']:
                img = data.float().cuda()
            elif model_type in ['fcn']:
                img = data.float().cuda()
                img = img.view(img.shape[0], -1)

            # ===================forward=====================
            output = model(img)
            if model_type in ['vae']:
                loss = loss_vae(output[0], img, output[1], output[2], criterion)
            else:
                loss = criterion(output, img)
            
            tot_loss.append(loss.item())

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ===================save_best====================
        mean_loss = np.mean(tot_loss)
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model, 'best_{}.pt'.format(train_name))
        # ===================log========================
        qqdm_train.set_infos({
        'epoch': f'{epoch + 1:.0f}/{num_epochs:.0f}',
        'loss': f'{mean_loss:.4f}',
        })

        # warm up
        scheduler.step()
        realLearningRate = scheduler.get_last_lr()[0]

        # wandb
        wandb.log({'epoch': epoch + 1, 'train_loss': mean_loss, 'learningRate':realLearningRate})
        # ===================save_last========================
        torch.save(model, 'last_{}.pt'.format(train_name))

def test(batch_size, test_dataloader, model_type):
    test = np.load(test_path, allow_pickle=True)

    eval_loss = nn.MSELoss(reduction='none')
    # load trained model
    checkpoint_path = 'best_'+ train_name +'.pt'
    model = torch.load(checkpoint_path)
    model.eval()
    # prediction file 
    out_file = train_name + '_predict.csv'

    anomality = list()
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            if model_type in ['cnn', 'vae', 'resnet']:
                img = data.float().cuda()
            elif model_type in ['fcn']:
                img = data.float().cuda()
                img = img.view(img.shape[0], -1)
            else:
                img = data[0].cuda()

            output = model(img)

            if model_type in ['cnn', 'resnet', 'fcn']:
                output = output
            elif model_type in ['res_vae']:
                output = output[0]
            elif model_type in ['vae']: # , 'vqvae'
                output = output[0]
            
            if model_type in ['fcn']:
                loss = eval_loss(output, img).sum(-1)
            else:
                loss = eval_loss(output, img).sum([1, 2, 3])

            anomality.append(loss)
    anomality = torch.cat(anomality, axis=0)
    anomality = torch.sqrt(anomality).reshape(len(test), 1).cpu().numpy()

    df = pd.DataFrame(anomality, columns=['Predicted'])
    df.to_csv(out_file, index_label = 'Id')

def main():
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    model_type = config['model_type']
    learning_rate = config['learning_rate']

    train_dataloader, test_dataloader = prep_dataloader(batch_size)

    train(num_epochs, learning_rate, train_dataloader, model_type)

    test(batch_size, test_dataloader, model_type)

if __name__ == '__main__':
    main()






# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
 
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset, DataLoader, Subset

import pandas as pd
import math

# use this to record my loss
import wandb

# Data Process
"""
The data is suitible for `torchvision.ImageFolder`. You can create a dataset with `torchvision.ImageFolder`. 
Details for image augmentation please refer to the comments in the following codes.
"""

torch.cuda.set_device(1)

train_path = 'real_or_drawing/train_data'
test_path = 'real_or_drawing/test_data'
train_name = '2k_strong_semi'

# hyperparameter
default_config = dict(
    train_batch=32,
    test_batch=128,
    num_epochs=10,
    learning_rate=1e-3,
    lamb = 0.0,
    do_semi = False,
    above_epoch = 3,

    early_stop=200,
    warm_up_epochs=5
)

# Initialize 
wandb.init(project='hw11', entity='nekokiku', config=default_config, name=train_name)
config = wandb.config


def data_process(train_batch, test_batch):

    source_transform = transforms.Compose([
        # Turn RGB to grayscale. (Bacause Canny do not support RGB images.)
        transforms.Grayscale(),
        # cv2 do not support skimage.Image, so we transform it to np.array, 
        # and then adopt cv2.Canny algorithm.
        transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
        # Transform np.array back to the skimage.Image.
        transforms.ToPILImage(),
        # 50% Horizontal Flip. (For Augmentation)
        transforms.RandomHorizontalFlip(),
        # Rotate +- 15 degrees. (For Augmentation), and filled with zero 
        # if there's empty pixel after rotation.
        transforms.RandomRotation(15, fill=(0,)),
        # Transform to tensor for model inputs.
        transforms.ToTensor(),
    ])

    target_transform = transforms.Compose([
        # Turn RGB to grayscale.
        transforms.Grayscale(),
        # Resize: size of source data is 32x32, thus we need to 
        #  enlarge the size of target data from 28x28 to 32x32。
        transforms.Resize((32, 32)),
        # 50% Horizontal Flip. (For Augmentation)
        transforms.RandomHorizontalFlip(),
        # Rotate +- 15 degrees. (For Augmentation), and filled with zero 
        # if there's empty pixel after rotation.
        transforms.RandomRotation(15, fill=(0,)),
        # Transform to tensor for model inputs.
        transforms.ToTensor(),
    ])

    source_dataset = ImageFolder(train_path, transform=source_transform)
    target_dataset = ImageFolder(test_path, transform=target_transform)

    source_dataloader = DataLoader(source_dataset, batch_size=train_batch, num_workers=5, shuffle=True, pin_memory=True)
    target_dataloader = DataLoader(target_dataset, batch_size=train_batch, num_workers=5, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(target_dataset, batch_size=test_batch, num_workers=5, shuffle=False, pin_memory=True)

    return source_dataset, target_dataset, source_dataloader, target_dataloader, test_dataloader

"""# Model
Feature Extractor: Classic VGG-like architecture
Label Predictor / Domain Classifier: Linear models.
"""

class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv(x).squeeze()
        return x

class LabelPredictor(nn.Module):

    def __init__(self):
        super(LabelPredictor, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 10),
        )

    def forward(self, h):
        c = self.layer(h)
        return c

class DomainClassifier(nn.Module):

    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),
        )

    def forward(self, h):
        y = self.layer(h)
        return y

class SemiDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.length = len(data)

    def __getitem__(self, idx):
        label = self.label[idx]
        data = self.data[idx]
        return data, label

    def __len__(self):
        return self.length


def get_pseudo_labels(test_dataloader, threshold=0.6):

    # load trained model
    label_predictor_path = 'model/predictor_'+ train_name +'.pt'
    feature_extractor_path = 'model/extractor_'+ train_name +'.pt'

    label_predictor = LabelPredictor().cuda()
    feature_extractor = FeatureExtractor().cuda()

    label_predictor.load_state_dict(torch.load(label_predictor_path))
    feature_extractor.load_state_dict(torch.load(feature_extractor_path))

    label_predictor.eval()
    feature_extractor.eval()

    # Define softmax function.
    softmax = nn.Softmax(dim=-1)

    fake_imgs = []
    fake_labels = []

    for i, (test_data, _) in enumerate(test_dataloader):
        # test_data is img
        with torch.no_grad():
            test_data = test_data.cuda()
            feature = feature_extractor(test_data)
            class_logits = label_predictor(feature)

        probs = softmax(class_logits)
        # Filter the data and construct a new dataset.
        values, indices = torch.max(probs, dim = -1)
        for j in range(len(indices)):
            if values[j].item() >= threshold:
                fake_imgs.append(test_data[j])
                fake_labels.append(indices[j].item())

    semi_data = SemiDataset(fake_imgs, fake_labels)

    label_predictor.train()
    feature_extractor.train()
    return semi_data

def train(learning_rate, num_epochs, source_dataset, source_dataloader, target_dataloader, test_dataloader, lamb, do_semi):

    '''
      Args:
        source_dataloader: source data的dataloader
        target_dataloader: target data的dataloader
        lamb: control the balance of domain adaptatoin and classification.
    '''

    feature_extractor = FeatureExtractor().cuda()
    label_predictor = LabelPredictor().cuda()
    domain_classifier = DomainClassifier().cuda()

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    optimizer_F = optim.Adam(feature_extractor.parameters(), lr=learning_rate)
    optimizer_C = optim.Adam(label_predictor.parameters(), lr=learning_rate)
    optimizer_D = optim.Adam(domain_classifier.parameters(), lr=learning_rate)

    # learning rate schedule
    # warm_up_with_cosine_lr = lambda epoch: epoch / config['warm_up_epochs'] if epoch <= config['warm_up_epochs'] else 0.5 * ( math.cos((epoch - config['warm_up_epochs']) /(num_epochs - config['warm_up_epochs']) * math.pi) + 1)
    # scheduler = torch.optim.lr_scheduler.LambdaLR( optimizer, lr_lambda=warm_up_with_cosine_lr)

    for epoch in range(num_epochs):
        # set the lambda
        lamb = lamb + 1/num_epochs
        lambd = 4. / (1. + math.exp(-10 * lamb)) - 2

        # D loss: Domain Classifier的loss
        # F loss: Feature Extrator & Label Predictor的loss
        running_D_loss, running_F_loss = 0.0, 0.0
        total_hit, total_num = 0.0, 0.0

        feature_extractor.train()
        label_predictor.train()
        domain_classifier.train()

        # train pseudo label
        if do_semi and epoch > config['above_epoch']:
            print("using pseudo labelling")
            pseudo_set = get_pseudo_labels(test_dataloader)
            concat_dataset = ConcatDataset([source_dataset, pseudo_set])
            concat_dataloader = DataLoader(concat_dataset, batch_size=config['train_batch'], shuffle=True, pin_memory=True)
           


        total_i = 0
        for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):

            source_data = source_data.cuda()
            source_label = source_label.cuda()
            target_data = target_data.cuda()

            # Mixed the source data and target data, or it'll mislead the running params
            #   of batch_norm. (runnning mean/var of soucre and target data are different.)
            mixed_data = torch.cat([source_data, target_data], dim=0)
            domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
            # set domain label of source data to be 1.
            domain_label[:source_data.shape[0]] = 1

            # Step 1 : train domain classifier
            feature = feature_extractor(mixed_data)
            # We don't need to train feature extractor in step 1.
            # Thus we detach the feature neuron to avoid backpropgation.
            domain_logits = domain_classifier(feature.detach())
            loss = domain_criterion(domain_logits, domain_label)
            running_D_loss+= loss.item()
            loss.backward()
            optimizer_D.step()



            # Step 2 : train feature extractor and label classifier
            class_logits = label_predictor(feature[:source_data.shape[0]])
            domain_logits = domain_classifier(feature)
            # loss = cross entropy of classification - lamb * domain binary cross entropy.
            #  The reason why using subtraction is similar to generator loss in disciminator of GAN
            loss = class_criterion(class_logits, source_label) - lambd * domain_criterion(domain_logits, domain_label)
            running_F_loss+= loss.item()
            loss.backward()
            optimizer_F.step()
            optimizer_C.step()

            optimizer_D.zero_grad()
            optimizer_F.zero_grad()
            optimizer_C.zero_grad()

            total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
            total_num += source_data.shape[0]
            print(i, end='\r')
            total_i = i

        train_D_loss, train_F_loss, train_acc = running_D_loss / (i+1), running_F_loss / (i+1), total_hit / total_num

        torch.save(feature_extractor.state_dict(), 'model/extractor_{}.pt'.format(train_name))
        torch.save(label_predictor.state_dict(), 'model/predictor_{}.pt'.format(train_name))
        print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}, total_iteration {:>3d}'.format(epoch, train_D_loss, train_F_loss, train_acc, total_i))

        # wandb
        wandb.log({'epoch': epoch + 1, 'train_D_loss': train_D_loss, 'train_F_loss':train_F_loss, 'train_acc':train_acc, 'lambd':lambd})

def predict(test_dataloader):

    # load trained model
    label_predictor_path = 'model/predictor_'+ train_name +'.pt'
    feature_extractor_path = 'model/extractor_'+ train_name +'.pt'

    label_predictor = LabelPredictor().cuda()
    feature_extractor = FeatureExtractor().cuda()

    label_predictor.load_state_dict(torch.load(label_predictor_path))
    feature_extractor.load_state_dict(torch.load(feature_extractor_path))

    label_predictor.eval()
    feature_extractor.eval()

    result = []
    with torch.no_grad():
        for i, (test_data, _) in enumerate(test_dataloader):
            test_data = test_data.cuda()

            feature = feature_extractor(test_data)

            class_logits = label_predictor(feature)

            x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
            result.append(x)

    result = np.concatenate(result)

    # Generate your submission
    df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
    df.to_csv('result/' + train_name + '_submission.csv',index=False)

def main():
    train_batch = config['train_batch']
    test_batch = config['test_batch']
    lamb = config['lamb']
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    do_semi = config['do_semi']

    print("loading data")
    source_dataset, target_dataset, source_dataloader, target_dataloader, test_dataloader = data_process(train_batch, test_batch)
    print("begin trainning")
    train(learning_rate, num_epochs, source_dataset, source_dataloader, target_dataloader, test_dataloader, lamb, do_semi)
    print("begin testing")
    predict(test_dataloader)
    print("Done!!!!!!")


if __name__ == '__main__':
    main()

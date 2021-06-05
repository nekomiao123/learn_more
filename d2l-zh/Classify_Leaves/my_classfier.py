import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import math
import matplotlib.pyplot as plt
import torchvision.models as models
# efficientnet
from efficientnet_pytorch import model as enet
# This is for the progress bar.
from tqdm import tqdm
# use this to record my loss
import wandb

import timm
import ttach as tta
# accelerate
# from apex import amp

# Specify the graphics card
torch.cuda.set_device(4)

#check device
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()

labels_dataframe = pd.read_csv('leaves_data/train.csv')
# Create list of alphabetically sorted labels.
leaves_labels = sorted(list(set(labels_dataframe['label'])))
n_classes = len(leaves_labels)
#Map each label string to an integer label.
class_to_num = dict(zip(leaves_labels, range(n_classes)))
num_to_class = {v : k for k, v in class_to_num.items()}

train_path = 'leaves_data/train.csv'
test_path = 'leaves_data/test.csv'
# we already have the iamges floder in the csv file，so we don't need it here
img_path = 'leaves_data/'
train_name = 'tf_efficientnet_b5_ns'

# hyperparameter
default_config = dict(
    batch_size=16,
    num_epoch=200,
    learning_rate=3e-4,             # learning rate of Adam
    weight_decay=0.001,             # weight decay 

    warm_up_epochs=10,
    model_path='./model/'+train_name+'_model.ckpt',
    saveFileName='./result/'+train_name+'_pred.csv',
    num_workers=5,
    model_name='effnetv2',
)

wandb.init(project='leaves_classfier', entity='nekokiku', config=default_config, name=train_name)
config = wandb.config

# my onw dataset
class LeavesData(Dataset):
    def __init__(self, csv_path, file_path, mode='train', valid_ratio=0.2, resize_height=256, resize_width=256):
        """
        Args:
            csv_path (string): csv file path
            img_path (string): image file path 
        """
        # we need resize our images
        self.resize_height = resize_height
        self.resize_width = resize_width

        self.file_path = file_path
        self.mode = mode

        # read the csv file using pandas
        self.data_info = pd.read_csv(csv_path, header=None)  # header=None, we can ingore the head
        # length
        self.data_len = len(self.data_info.index) - 1
        self.train_len = int(self.data_len * (1 - valid_ratio))

        if mode == 'train':
            # The first column is our image file name
            self.train_image = np.asarray(self.data_info.iloc[1:self.train_len, 0]) 
            # The second colimn is the label
            self.train_label = np.asarray(self.data_info.iloc[1:self.train_len, 1])
            self.image_arr = self.train_image 
            self.label_arr = self.train_label
        elif mode == 'valid':
            self.valid_image = np.asarray(self.data_info.iloc[self.train_len:, 0])  
            self.valid_label = np.asarray(self.data_info.iloc[self.train_len:, 1])
            self.image_arr = self.valid_image
            self.label_arr = self.valid_label
        elif mode == 'test':
            self.test_image = np.asarray(self.data_info.iloc[1:, 0])
            self.image_arr = self.test_image
            
        self.real_len = len(self.image_arr)

        print('Finished reading the {} set of Leaves Dataset ({} samples found)'
              .format(mode, self.real_len))

    def __getitem__(self, index):
        # we can get the file name
        single_image_name = self.image_arr[index]

        # read our image
        img_as_img = Image.open(self.file_path + single_image_name)

        # transform
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),   # Horizontal random flip
                transforms.RandomVerticalFlip(p=0.5),     # Vertical random flip
                transforms.RandomPerspective(p=0.5),
                transforms.RandomAffine(35),
                transforms.RandomRotation(45),            # random rotation
                transforms.RandomGrayscale(p=0.025),      #概率转换成灰度率，3通道就是R=G=B
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # mean，标准差
            ])
        else:
            # we don't need transfrom for valid and test
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # mean，标准差
            ])

        img_as_img = transform(img_as_img)

        if self.mode == 'test':
            return img_as_img
        else:
            # string label
            label = self.label_arr[index]
            # number label
            number_label = class_to_num[label]

            return img_as_img, number_label  # image and label

    def __len__(self):
        return self.real_len

# efficientnet
class enetv2(nn.Module):
    def __init__(self, backbone, out_dim):
        super(enetv2, self).__init__()
        self.enet = enet.EfficientNet.from_pretrained(backbone)
        self.myfc = nn.Linear(self.enet._fc.in_features, out_dim)
        self.enet._fc = nn.Identity()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=3, bias=False)

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.conv1(x)
        x = self.extract(x)
        x = self.myfc(x)
        return x

# BNNeck
class res50(torch.nn.Module):
    def __init__(self, num_classes):
        super(res50, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = torch.nn.Sequential(
                        resnet.conv1,
                        resnet.bn1,
                        resnet.relu,
                        resnet.layer1,
                        resnet.layer2,
                        resnet.layer3,
                        resnet.layer4
        )
        self.pool = torch.nn.AdaptiveMaxPool2d(1)
        self.bnneck = nn.BatchNorm1d(2048)
        self.bnneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(2048, num_classes, bias=False)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        feat = x.view(x.shape[0], -1)
        feat = self.bnneck(feat)
        if not self.training:
            return nn.functional.normalize(feat, dim=1, p=2)
        x = self.classifier(feat)
        return x

# dual pooling
class res18(nn.Module):
    def __init__(self, num_classes):
        super(res18, self).__init__()
        self.base = models.resnet34(pretrained=True)
        self.feature = nn.Sequential(
            self.base.conv1,
            self.base.bn1,
            self.base.relu,
            self.base.maxpool,
            self.base.layer1,
            self.base.layer2,
            self.base.layer3,
            self.base.layer4
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.reduce_layer = nn.Conv2d(1024, 512, 1)
        self.fc  = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
            )
    def forward(self, x):
        bs = x.shape[0]
        x = self.feature(x)
        x1 = self.avg_pool(x)
        x2 = self.max_pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.reduce_layer(x).view(bs, -1)
        logits = self.fc(x)
        return logits

def pre_data(batch_size, num_workers):
    train_dataset = LeavesData(train_path, img_path, mode='train')
    val_dataset = LeavesData(train_path, img_path, mode='valid')
    test_dataset = LeavesData(test_path, img_path, mode='test')
    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers
        )
    val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers
        )
    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers
        )
    
    return train_loader, val_loader, test_loader

# frozen the layers or not 
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.requires_grad = False

# here we use resnet18
def res_model(num_classes, feature_extract = False, use_pretrained=True):

    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))

    return model_ft

def initialize_model(num_classes, model_name = config['model_name'], feature_extract = False, use_pretrained=True):
    # 选择合适的模型，不同模型的初始化方法稍微有点区别
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))
        input_size = 224

    elif model_name == 'dualres':
        """ Dual pooling Resnet18
        """
        model_ft = res18(num_classes)

    elif model_name == 'effnet':
        """ efficientnet
        """
        model_ft = enetv2(backbone='efficientnet-b2',out_dim=num_classes)

    elif model_name == 'effnetv2':
        """ efficientnetv2
        """
        print("using efficientnetv2")
        model_ft = timm.create_model('tf_efficientnet_b5_ns', pretrained=True, num_classes=num_classes)

    elif model_name == 'seresnext':
        """ seresnext-50
        """
        print("using se-resnext-50")
        model_ft = timm.create_model('seresnext50_32x4d', pretrained=True, num_classes=num_classes)

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs,num_classes))
        input_size = 224

    elif model_name == "vgg":
        """ VGG19_bn
        """
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs,num_classes))
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft

class LabelSmoothCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, label, smoothing=0.1):
        pred = F.softmax(pred, dim=1)
        one_hot_label = F.one_hot(label, pred.size(1)).float()
        smoothed_one_hot_label = (
            1.0 - smoothing) * one_hot_label + smoothing / pred.size(1)
        loss = (-torch.log(pred)) * smoothed_one_hot_label
        loss = loss.sum(axis=1, keepdim=False)
        loss = loss.mean()
        return loss

def train(train_loader, val_loader, num_epoch, learning_rate, weight_decay, model_path):

    # Initialize a model, and put it on the device specified.
    model = initialize_model(num_classes=176)
    model = model.to(device)
    model.device = device
    # For the classification task, we use cross-entropy as the measurement of performance.
    criterion = LabelSmoothCELoss()

    # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)

    # learning rate schedule
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    # warm_up_with_cosine_lr
    warm_up_with_cosine_lr = lambda epoch: epoch / config['warm_up_epochs'] if epoch <= config['warm_up_epochs'] else 0.5 * ( math.cos((epoch - config['warm_up_epochs']) /(num_epoch - config['warm_up_epochs']) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR( optimizer, lr_lambda=warm_up_with_cosine_lr)

    # The number of training epochs.
    n_epochs = num_epoch
    best_acc = 0.0
    for epoch in range(n_epochs):
        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train() 
        # These are used to record information in training.
        train_loss = []
        train_accs = []
        # Iterate the training set by batches.
        for batch in tqdm(train_loader):
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs)
            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels)
            
            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()
            # Compute the gradients for parameters.
            loss.backward()
            # Update the parameters with computed gradients.
            optimizer.step()
            
            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)

        # The average loss and accuracy of the training set is the average of the recorded values.
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    
        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()
        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []
        # Iterate the validation set by batches.
        for batch in tqdm(val_loader):
            imgs, labels = batch
            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))
                
            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels.to(device))

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # learning rate decay and print 
        scheduler.step()
        realLearningRate = scheduler.get_last_lr()[0]
        # wandb
        wandb.log({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': valid_loss, 'train_acc':train_acc, 'valid_acc':valid_acc, 'LearningRate':realLearningRate})

        # if the model improves, save a checkpoint at this epoch
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), model_path)
            print('saving model with acc {:.3f}'.format(best_acc))

def predict(model_path, test_loader, saveFileName, iftta):

    ## predict
    model = initialize_model(num_classes=176)

    # create model and load weights from checkpoint
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))

    if iftta:
        print("Using TTA")
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[0, 180]),
                # tta.Scale(scales=[1, 0.3]), 
            ]
        )
        model = tta.ClassificationTTAWrapper(model, transforms)

    # Make sure the model is in eval mode.
    # Some modules like Dropout or BatchNorm affect if the model is in training mode.
    model.eval()
    
    # Initialize a list to store the predictions.
    predictions = []
    # Iterate the testing set by batches.
    for batch in tqdm(test_loader):

        imgs = batch
        with torch.no_grad():
            logits = model(imgs.to(device))
        
        # Take the class with greatest logit as prediction and record it.
        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    preds = []
    for i in predictions:
        preds.append(num_to_class[i])

    test_data = pd.read_csv('leaves_data/test.csv')
    test_data['label'] = pd.Series(preds)
    submission = pd.concat([test_data['image'], test_data['label']], axis=1)
    submission.to_csv(saveFileName, index=False)
    print("Done!!!!!!!!!!!!!!!!!!!!!!!!!!!")

def main():
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    num_epoch = config['num_epoch']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    model_path = config['model_path']
    saveFileName = config['saveFileName']

    print("loading data")
    train_loader, val_loader, test_loader = pre_data(batch_size, num_workers)
    print("training")
    train(train_loader, val_loader, num_epoch, learning_rate, weight_decay, model_path)
    print("testing")
    predict(model_path, test_loader, saveFileName, True)

if __name__ == '__main__':
    main()

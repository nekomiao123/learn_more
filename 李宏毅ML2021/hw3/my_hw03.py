# -*- coding: utf-8 -*-

"""
# **Homework 3 - Convolutional Neural Network**

This is the example code of homework 3 of the machine learning course by Prof. Hung-yi Lee.

In this homework, you are required to build a convolutional neural network for image classification, possibly with some advanced training tips.


There are three levels here:

**Easy**: Build a simple convolutional neural network as the baseline. (2 pts)

**Medium**: Design a better architecture or adopt different data augmentations to improve the performance. (2 pts)

**Hard**: Utilize provided unlabeled data to obtain better results. (2 pts)

## **About the Dataset**

The dataset used here is food-11, a collection of food images in 11 classes.

For the requirement in the homework, TAs slightly modified the data.
Please DO NOT access the original fully-labeled training data or testing labels.

Also, the modified dataset is for this course only, and any further distribution or commercial use is forbidden.

I just change the .ipynb to .py file.
"""

# Import necessary packages.
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset
import torchvision.models as models
# This is for the progress bar.
from tqdm import tqdm

# tensorboard
from torch.utils.tensorboard import SummaryWriter

# Specify the graphics card
torch.cuda.set_device(3)

# hyper-parameters 
config = {
    'learning_rate': 0.0003,       # learning rate
    'batch_size': 32,
    'num_epoch': 200,
    'above_epoch': 0,    
    'weight_decay': 1e-4,
    'model_name': 'vgg',
    'model_path': './train_model/vgg_11_model.ckpt',
    'warm_up_epochs': 5
}

saveFileName = config['model_name'] + '_predict.csv'
comment = '_' + config['model_name'] + '_32_schedule_warm_up'
writer = SummaryWriter(comment=comment)

class UlabelDataset(Dataset):
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


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # input image size: [3, 128, 128]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 11)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x = self.cnn_layers(x)

        # The extracted feature map must be flatten before going to fully-connected layers.
        x = x.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(num_classes, model_name=config['model_name'], feature_extract = False, use_pretrained=False):
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
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
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

    return model_ft, input_size


def prep_dataloader(batch_size, data_root='./food-11/'):

    # It is important to do data augmentation in training.
    # However, not every augmentation is useful.
    # Please think about what kind of augmentation is helpful for food recognition.
    train_tfm = transforms.Compose([
        # Resize the image into a fixed shape (height = width = 128)
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),#从中心开始裁剪
        transforms.RandomHorizontalFlip(p=0.5),   #随机水平翻转 选择一个概率
        transforms.RandomRotation(45),            #随机旋转，-45到45度之间随机选
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),#参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
        transforms.RandomGrayscale(p=0.025),      #概率转换成灰度率，3通道就是R=G=B
        # You may add some transforms here.
        # ToTensor() should be the last one of the transforms.
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#均值，标准差
    ])

    # We don't need augmentations in testing and validation.
    # All we need here is to resize the PIL image and transform it into Tensor.
    test_tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),#从中心开始裁剪
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#均值，标准差
    ])

    # Construct datasets.
    # The argument "loader" tells how torchvision reads the data.
    train_set_path = data_root + "training/labeled"
    valid_set_path = data_root + "validation"
    unlabeled_set_path = data_root + "training/unlabeled"
    test_set_path = data_root + "testing"
    train_set = DatasetFolder(train_set_path, loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
    valid_set = DatasetFolder(valid_set_path, loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
    unlabeled_set = DatasetFolder(unlabeled_set_path, loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
    test_set = DatasetFolder(test_set_path, loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)

    # Construct data loaders.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


    # dataiter = iter(train_loader)
    # inputs, classes = dataiter.next()
    # print("train inputs")
    # print(inputs)
    # print("train class")
    # print(classes)


    return train_loader, valid_loader, test_loader, train_set, valid_set, unlabeled_set, test_set

#check device
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

# The function "get_pseudo_labels" is used for semi-supervised learning.
def get_pseudo_labels(dataset, model, threshold=0.65):
    # This functions generates pseudo-labels of a dataset using given model.
    # It returns an instance of DatasetFolder containing images whose prediction confidences exceed a given threshold.
    # You are NOT allowed to use any models trained on external data for pseudo-labeling.
    device = get_device()

    # Make sure the model is in eval mode.
    model.eval()
    # Define softmax function.
    softmax = nn.Softmax(dim=-1)

    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=5, pin_memory=True)

    imgs = []
    labels = []
    # Iterate over the dataset by batches.
    for batch in tqdm(dataloader):
        img, _ = batch

        # Forward the data
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img.to(device))

        # Obtain the probability distributions by applying softmax on logits.
        probs = softmax(logits)

        # ---------- TODO ----------
        # Filter the data and construct a new dataset.
        tem_label = []
        datalabel = probs.argmax(dim=-1)
        imgs.extend(img)
        labels.extend(datalabel.cpu().numpy().tolist())

    SemiData = UlabelDataset(imgs, labels)
    dataset = SemiData

    # dataloader2 = DataLoader(SemiData, batch_size=2, shuffle=False, num_workers=3, pin_memory=True)
    # dataiter = iter(dataloader2)
    # inputs, classes = dataiter.next()
    # print("inputs")
    # print(inputs)
    # print("class")
    # print(classes)

    # Turn off the eval mode.
    model.train()
    return dataset

def train_data(train_loader, valid_loader, train_set, unlabeled_set, model_path, device, learning_rate, weight_decay, num_epoch, batch_size, do_semi = False):

    # "cuda" only when GPUs are available.
    device = device

    # use resnet18
    model, inputs = initialize_model(num_classes=11)

    # Initialize a model, and put it on the device specified.
    model = model.to(device)
    model.device = device

    # For the classification task, we use cross-entropy as the measurement of performance.
    criterion = nn.CrossEntropyLoss()

    # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)

    # learning rate schedule
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    # warm_up_with_cosine_lr
    warm_up_with_cosine_lr = lambda epoch: epoch / config['warm_up_epochs'] if epoch <= config['warm_up_epochs'] else 0.5 * ( math.cos((epoch - config['warm_up_epochs']) /(num_epoch - config['warm_up_epochs']) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR( optimizer, lr_lambda=warm_up_with_cosine_lr)

    # The number of training epochs.
    n_epochs = num_epoch

    # Whether to do semi-supervised learning.
    do_semi = do_semi

    best_acc = 0.0
    for epoch in range(n_epochs):
        # ---------- TODO ----------
        # In each epoch, relabel the unlabeled dataset for semi-supervised learning.
        # Then you can combine the labeled dataset and pseudo-labeled dataset for the training.
        if do_semi and epoch > config['above_epoch']:
            # Obtain pseudo-labels for unlabeled data using trained model.
            pseudo_set = get_pseudo_labels(unlabeled_set, model)

            # Construct a new dataset and a data loader for training.
            # This is used in semi-supervised learning only.
            concat_dataset = ConcatDataset([train_set, pseudo_set])
            train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)

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

            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs.to(device))

            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels.to(device))

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            
            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)
        
        # The average loss and accuracy of the training set is the average of the recorded values.
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Loss/train", train_loss, epoch)

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []

        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader):
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
        
        writer.add_scalar("Loss/Val", valid_loss, epoch)
        writer.add_scalar("Acc/Val", valid_acc, epoch)

        # learning rate decay and print 
        scheduler.step()
        realLearningRate = scheduler.get_last_lr()[0]
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] real learning rate = {realLearningRate:.5f}")
        # realLearningRate = optimizer.state_dict()['param_groups'][0]['lr']
        writer.add_scalar("LearningRate", realLearningRate, epoch)

        # if the model improves, save a checkpoint at this epoch
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), model_path)

            print('saving model with acc {:.3f}'.format(best_acc))
        
    writer.flush()
    writer.close()

def predict_data(test_loader, model_path, device):

    # use resnet18
    model, inputs = initialize_model(num_classes=11)

    # create model and load weights from checkpoint
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))

    # Make sure the model is in eval mode.
    # Some modules like Dropout or BatchNorm affect if the model is in training mode.
    model.eval()

    # Initialize a list to store the predictions.
    predictions = []

    # Iterate the testing set by batches.
    for batch in tqdm(test_loader):
        # A batch consists of image data and corresponding labels.
        # But here the variable "labels" is useless since we do not have the ground-truth.
        # If printing out the labels, you will find that it is always 0.
        # This is because the wrapper (DatasetFolder) returns images and labels for each batch,
        # so we have to create fake labels to make it work normally.
        imgs, labels = batch


        # We don't need gradient in testing, and we don't even have labels to compute loss.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # Take the class with greatest logit as prediction and record it.
        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    # Save predictions into the file.
    with open(saveFileName, "w") as f:

        # The first row must be "Id, Category"
        f.write("Id,Category\n")

        # For the rest of the rows, each image id corresponds to a predicted class.
        for i, pred in  enumerate(predictions):
            f.write(f"{i},{pred}\n")

def main():
    print('------------------------------------------------------------------------------------------------------')
    device = get_device()
    print(f'DEVICE: {device}')
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    num_epoch = config['num_epoch']     
    model_path = config['model_path']
    weight_decay = config['weight_decay']

    train_loader, valid_loader, test_loader, train_set, valid_set, unlabeled_set, test_set = prep_dataloader(batch_size)
    print('New train begins')
    train_data(train_loader, valid_loader, train_set, unlabeled_set, model_path, device, learning_rate, weight_decay, num_epoch, batch_size)
    predict_data(test_loader, model_path, device)
    print('All done')

if __name__ == '__main__':
    main()

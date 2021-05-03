# -*- coding: utf-8 -*-

import random
import torch
import numpy as np

"""## Import Packages
First, we need to import packages that will be used later.
"""
import os
import glob

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from qqdm import qqdm
from tqdm import tqdm


# You may replace the workspace directory if you want.
workspace_dir = '.'
# Specify the graphics card
torch.cuda.set_device(4)

""" Medium: WGAN, 50 epoch, n_critic=5, clip_value=0.01 """
# hyper-parameters 
config = {
    'learning_rate': 1e-4,
    'batch_size': 64,
    'z_dim':100,
    'num_epoch': 50,
    'weight_decay': 1e-3,
    'n_critic':5,
    'clip_value':0.01,
    'warm_up_epochs': 5
}

# Training hyperparameters
z_sample = Variable(torch.randn(100, config['z_dim'])).cuda()


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

same_seeds(2021)

"""## Dataset
1. Resize the images to (64, 64)
1. Linearly map the values from [0, 1] to  [-1, 1].
"""

class CrypkoDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self,idx):
        fname = self.fnames[idx]
        # 1. Load the image
        img = torchvision.io.read_image(fname)
        # 2. Resize and normalize the images using torchvision.
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples


def get_dataset(root):
    fnames = glob.glob(os.path.join(root, '*'))
    # 1. Resize the image to (64, 64)
    # 2. Linearly map [0, 1] to [-1, 1]
    compose = [
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(compose)
    dataset = CrypkoDataset(fnames, transform)
    return dataset


"""## Model
Here, we use DCGAN as the model structure. Feel free to modify your own model structure.

Note that the `N` of the input/output shape stands for the batch size.
"""

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    """
    Input shape: (N, in_dim)
    Output shape: (N, 3, 64, 64)
    """
    def __init__(self, in_dim, dim=64):
        super(Generator, self).__init__()
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU()
        )
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y


class Discriminator(nn.Module):
    """
    Input shape: (N, 3, 64, 64)
    Output shape: (N, )
    """
    def __init__(self, in_dim, dim=64):
        super(Discriminator, self).__init__()

        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2),
            )
            
        """ Medium: Remove the last sigmoid layer for WGAN. """
        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2), 
            nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim * 2),
            conv_bn_lrelu(dim * 2, dim * 4),
            conv_bn_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4),
            # nn.Sigmoid(), 
        )
        self.apply(weights_init)
        
    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y


def prep_dataloader(batch_size, num_wokers=4, root='faces'):
    # Dataset
    dataset = get_dataset(os.path.join(workspace_dir, root))
    # DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_wokers)

    log_dir = os.path.join(workspace_dir, 'w_logs')
    ckpt_dir = os.path.join(workspace_dir, 'w_checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    return dataset, dataloader, log_dir, ckpt_dir


def show_the_data(dataset):
    images = [(dataset[i]+1)/2 for i in range(16)]
    grid_img = torchvision.utils.make_grid(images, nrow=4)
    plt.figure(figsize=(10,10))
    plt.imshow(grid_img.permute(1, 2, 0))

    # The remote server cannot display the picture
    # plt.show()
    # but we can save it 
    plt.savefig('pictures/example.jpg')


def train_data(z_dim, n_epoch, n_critic, lr, dataloader, log_dir, ckpt_dir, clip_value):
    # Model
    G = Generator(in_dim=z_dim).cuda()
    D = Discriminator(3).cuda()
    G.train()
    D.train()

    # Loss
    criterion = nn.BCELoss()

    """ Medium: Use RMSprop for WGAN. """
    # Optimizer
    # opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    # opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.RMSprop(D.parameters(), lr=lr)
    opt_G = torch.optim.RMSprop(G.parameters(), lr=lr)

    """### Training loop
    We store some pictures regularly to monitor the current performance of the Generator, and regularly record checkpoints.
    """
    steps = 0
    Loss_D_S = []
    Loss_G_S = []
    for e, epoch in enumerate(range(n_epoch)):
        progress_bar = tqdm(dataloader)
        for i, data in enumerate(progress_bar):
            imgs = data
            imgs = imgs.cuda()

            bs = imgs.size(0)

            # ============================================
            #  Train D
            # ============================================
            z = Variable(torch.randn(bs, z_dim)).cuda()
            r_imgs = Variable(imgs).cuda()
            f_imgs = G(z)

            """ Medium: Use WGAN Loss. """
            # # Label
            # r_label = torch.ones((bs)).cuda()
            # f_label = torch.zeros((bs)).cuda()

            # # Model forwarding
            # r_logit = D(r_imgs.detach())
            # f_logit = D(f_imgs.detach())
            
            # # Compute the loss for the discriminator.
            # r_loss = criterion(r_logit, r_label)
            # f_loss = criterion(f_logit, f_label)
            # loss_D = (r_loss + f_loss) / 2

            # WGAN Loss
            loss_D = -torch.mean(D(r_imgs)) + torch.mean(D(f_imgs))

            # Model backwarding
            D.zero_grad()
            loss_D.backward()

            # Update the discriminator.
            opt_D.step()

            """ Medium: Clip weights of discriminator. """
            for p in D.parameters():
               p.data.clamp_(-clip_value, clip_value)

            # ============================================
            #  Train G
            # ============================================
            if steps % n_critic == 0:
                # Generate some fake images.
                z = Variable(torch.randn(bs, z_dim)).cuda()
                f_imgs = G(z)

                # Model forwarding
                f_logit = D(f_imgs)

                """ Medium: Use WGAN Loss"""
                # Compute the loss for the generator.
                # loss_G = criterion(f_logit, r_label)
                # WGAN Loss
                loss_G = -torch.mean(D(f_imgs))

                # Model backwarding
                G.zero_grad()
                loss_G.backward()

                # Update the generator.
                opt_G.step()

            steps += 1

            Loss_D = round(loss_D.item(), 4)
            Loss_G = round(loss_G.item(), 4)
            Loss_D_S.append(Loss_D)
            Loss_G_S.append(Loss_G)

            Step  = steps
        

        D_loss = sum(Loss_D_S) / len(Loss_D_S)
        G_loss = sum(Loss_G_S) / len(Loss_G_S)

        print(f"[ Train | {epoch + 1:03d}/{n_epoch:03d} ] loss_D = {D_loss:.5f}, loss_G = {G_loss:.5f}")
        G.eval()
        f_imgs_sample = (G(z_sample).data + 1) / 2.0
        filename = os.path.join(log_dir, f'Epoch_{epoch+1:03d}.jpg')
        torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
        print(f' | Save some samples to {filename}.')
        
        # Show generated images in the jupyter notebook.
        # grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
        # plt.figure(figsize=(10,10))
        # plt.imshow(grid_img.permute(1, 2, 0))
        # plt.show()
        # G.train()

        if (e+1) % 5 == 0 or e == 0:
            # Save the checkpoints.
            torch.save(G.state_dict(), os.path.join(ckpt_dir, 'G.pth'))
            torch.save(D.state_dict(), os.path.join(ckpt_dir, 'D.pth'))

"""## Inference
Use the trained model to generate anime faces!
"""

def generate_data(z_dim, log_dir ,ckpt_dir):
    G = Generator(z_dim)
    G.load_state_dict(torch.load(os.path.join(ckpt_dir, 'G.pth')))
    G.eval()
    G.cuda()
    """### Generate and show some images."""

    # Generate 1000 images and make a grid to save them.
    n_output = 1000
    z_sample = Variable(torch.randn(n_output, z_dim)).cuda()
    imgs_sample = (G(z_sample).data + 1) / 2.0
    filename = os.path.join(log_dir, 'result.jpg')
    torchvision.utils.save_image(imgs_sample, filename, nrow=10)

    # Show 32 of the images.
    grid_img = torchvision.utils.make_grid(imgs_sample[:32].cpu(), nrow=10)
    plt.figure(figsize=(10,10))
    plt.imshow(grid_img.permute(1, 2, 0))

    # plt.show()
    plt.savefig('pictures/wgan_generated.jpg')


def main():
    z_dim = config['z_dim']
    n_epoch = config['num_epoch']
    n_critic = config['n_critic']
    lr = config['learning_rate']
    batch_size = config['batch_size']
    clip_value = config['clip_value']

    print("read data")
    dataset, dataloader, log_dir, ckpt_dir = prep_dataloader(batch_size)
    print("-----------------------begin trainning--------------------------")
    train_data(z_dim, n_epoch, n_critic, lr, dataloader, log_dir, ckpt_dir, clip_value)
    print("-----------------------begin generating--------------------------")
    generate_data(z_dim, log_dir, ckpt_dir)
    print("--------------------------Done-----------------------------------")
    # print("show the data")
    # show_the_data(dataset)


if __name__ == '__main__':
    main()
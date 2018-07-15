import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from dataset import ImageDataset

from models import Generator
from models import Discriminator
from models import weights_init

# < Default parameters >

start_epoch = 0
n_epochs = 200
lr = 0.0002
decay_epoch = 100
lr_decay = lr / (n_epochs-decay_epoch)

#=========================================>
#   < Defining variables >
#=========================================>

device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

# DataLoader:
transform = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = ImageDataset('./data/monet/', transform = transform, train=True)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, 
                                         shuffle=True)

# Networks:
G_A2B = Generator().to(device)
D_B = Discriminator().to(device)
G_B2A = Generator().to(device)
D_A = Discriminator().to(device)

G_A2B.apply(weights_init)
D_B.apply(weights_init)
G_B2A.apply(weights_init)
D_A.apply(weights_init)

# Loss functions:
loss_fn_GAN = nn.MSELoss()
loss_fn_cycle = nn.L1Loss()
loss_fn_identity = nn.L1Loss()

# Optimizers:
optimizer_G_A2B = optim.Adam(G_A2B.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(G_A2B.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_G_B2A = optim.Adam(G_B2A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_A = optim.Adam(G_B2A.parameters(), lr=lr, betas=(0.5, 0.999))

## Learning rate schedulers:
#lr_scheduler_G_A2B  = optim.lr_scheduler.LambdaLR(optimizer_G_A2B, 
#                        lr_lambda=)
#
#lr_scheduler_D_B  = optim.lr_scheduler.LambdaLR(optimizer_D_B, 
#                        lr_lambda=)
#
#lr_scheduler_G_B2A  = optim.lr_scheduler.LambdaLR(optimizer_G_B2A, 
#                        lr_lambda=)
#
#lr_scheduler_D_A  = optim.lr_scheduler.LambdaLR(optimizer_D_A, 
#                        lr_lambda=)

















































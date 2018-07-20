import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from dataset import ImageDataset

from models import Generator
from models import Discriminator
#from models import weights_init

########## Parameters ###########

batch_size = 1
start_epoch = 0
n_epochs = 200

lr = 0.0002
decay_epoch = 100
lr_decay = lr / (n_epochs-decay_epoch)

in_ch = 3
out_ch = 3

########## Defining variables ###########

device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

# DataLoader:
transform = transforms.Compose([transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = ImageDataset('./data/monet/', transform = transform, train=True)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                         shuffle=True)

# Networks:
G_A2B = Generator(in_ch, out_ch).to(device)
D_B = Discriminator(in_ch).to(device)
G_B2A = Generator(in_ch, out_ch).to(device)
D_A = Discriminator(in_ch).to(device)

#D_B.apply(weights_init)
#G_B2A.apply(weights_init)
#G_A2B.apply(weights_init)
#D_A.apply(weights_init)

# Loss functions:
loss_fn_GAN = nn.MSELoss()
loss_fn_cycle = nn.L1Loss()
loss_fn_identity = nn.L1Loss()

# variables on device:
real_label = torch.full((batch_size,1,1,1), 1, device=device)
fake_label = torch.full((batch_size,1,1,1), 0, device=device)

########## Training ###########

for epoch in range(start_epoch, n_epochs):
    
    # Optimzers:
    optimizer_G_A2B = optim.Adam(G_A2B.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(G_A2B.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_G_B2A = optim.Adam(G_B2A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(G_B2A.parameters(), lr=lr, betas=(0.5, 0.999))
    
    for i, data in enumerate(dataloader):
        
        real_A, real_B = data
        real_A, real_B = real_A.to(device), real_B.to(device)
        
        ########## Generator A-to-B ##########
        
        optimizer_G_A2B.zero_grad()
        
        # Identity loss:
        same_B = G_A2B(real_B)
        loss_identity = loss_fn_identity(same_B, real_B)
        
        # GAN loss:
        fake_B = G_A2B(real_A)
        D_score = D_B(fake_B)
        loss_GAN = loss_fn_GAN(D_score, real_label)
        
        # Cycle loss:
        recycled_A = G_B2A(fake_B)
        loss_cycle = loss_fn_cycle(recycled_A, real_A)
        
        # Total loss:
        loss_G_A2B = (5 * loss_identity) + loss_GAN + (10 * loss_cycle)
        loss_G_A2B.backward()
        
        optimizer_G_A2B.step()
        
        
        ########## Discriminator B ##########
        
        optimizer_D_B.zero_grad()
        
        # Real loss:
        D_score = D_B(real_B)
        loss_real = loss_fn_GAN(D_score, real_label)
        
        # Fake loss:
        fake_B = G_A2B(real_A)
        D_score = D_B(fake_B)
        loss_fake = loss_fn_GAN(D_score, fake_label)
        
        # Total loss:
        loss_D_B = (loss_real + loss_fake) / 2
        loss_D_B.backward()
        
        optimizer_D_B.step()
        
                       
        ########## Generator B-to-A ##########
        
        optimizer_G_B2A.zero_grad()
        
        # Identity loss:
        same_A = G_B2A(real_A)
        loss_identity = loss_fn_identity(same_A, real_A)
        
        # GAN loss:
        fake_A = G_B2A(real_B)
        D_score = D_A(fake_A)
        loss_GAN = loss_fn_GAN(D_score, real_label)
        
        # Cycle loss:
        recycled_B = G_A2B(fake_A)
        loss_cycle = loss_fn_cycle(recycled_B, real_B)
        
        # Total loss:
        loss_G_B2A = (5 * loss_identity) + loss_GAN + (10 * loss_cycle)
        loss_G_B2A.backward()
        
        optimizer_G_B2A.step()
        
        
        ########## Discriminator A ##########
        
        optimizer_D_A.zero_grad()
        
        # Real loss:
        D_score = D_A(real_A)
        loss_real = loss_fn_GAN(D_score, real_label)
        
        # Fake loss:
        fake_A = G_B2A(real_B)
        D_score = D_A(fake_A)
        loss_fake = loss_fn_GAN(D_score, fake_label)
        
        # Total loss:
        loss_D_A = (loss_real + loss_fake) / 2
        loss_D_A.backward()
        
        optimizer_D_A.step()
        
        print("HELLO")
        
    if(epoch>decay_epoch):
        lr -= lr_decay
    
    # Save the state
    
    
        
 

import torch
import torch.nn as nn

# DCGAN Weight init :
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

        
class Generator(nn.Module):
    
    def __init__(self, nz=100):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
                
            # input z, size: nz x 1 x 1
            nn.ConvTranspose2d( nz, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # state size: 512 x 4 x 4
            nn.ConvTranspose2d( 512, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # state size: 512 x 8 x 8
            nn.ConvTranspose2d( 512, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # state size: 512 x 32 x 32
            nn.ConvTranspose2d( 512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # state size: 256 x 64 x 64
            nn.ConvTranspose2d( 256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # state size: 128 x 128 x 128
            nn.ConvTranspose2d( 128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # state size: 64 x 256 x 256
            nn.ConvTranspose2d( 64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            
            # output size: 3 x 256 x 256
            )
    
    def forward(self, input):
        out = self.main(input)
        
        return out


class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            
            # input size: 3 x 256 x 256
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size: 64 x 128 x 128 
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size: 128 x 64 x 64 
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size: 256 x 32 x 32
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size: 512 x 16 x 16
            nn.Conv2d(512, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size: 512 x 8 x 8
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            
            # state size: 512 x 4 x 4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            
            # output size: 1 x 1 x 1
            )
    
    def forward(self, input):
        out = self.main(input)
        
        # out size: batch_size x 1 x 1 x 1
        out = out.view(-1, 1).squeeze(1)
        
        # out size: 1 x batch_size
        return out 



## TESTING CODE :
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#dis = Discriminator().to(device)
#
#gen = Generator().to(device)
#
#z = torch.randn(1,100,1,1).to(device)
#
#print(z.size())
#
#fake = gen(z)
#
#print(fake.size())
#
#print(dis(fake))

    


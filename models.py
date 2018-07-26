import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class ResidualBlock(nn.Module):
    def __init__(self, n_features):
        super(ResidualBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
                
                nn.Conv2d(n_features, n_features, 3, 1, 1),
                nn.BatchNorm2d(n_features),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(n_features, n_features, 3, 1, 1),
                nn.BatchNorm2d(n_features),
                nn.ReLU(inplace=True) )
    
    def forward(self, x):
        return self.conv_block(x) + x


class Generator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, n_blocks=4):
        super(Generator, self).__init__()
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_blocks = n_blocks
        
        # input size: in_ch x 256 x 256
        
        n_features = 64
        
        # 1st conv:
        model = [ nn.Conv2d(in_ch, n_features, 4, 2, 1),
                  nn.BatchNorm2d(n_features),
                  nn.ReLU(inplace=True) ]
        
        for i in range(self.n_blocks):
            model += [ResidualBlock(n_features)]
        
        # last conv with tanh
        model += [ nn.ConvTranspose2d(n_features, out_ch, 4, 2, 1),
                   nn.Tanh() ]
        
        # output size: out_ch x 256 x 256
        
        self.model = nn.Sequential(*model)
    
    def forward(self, input):
        return self.model(input)
        

class Discriminator(nn.Module):
    def __init__(self, in_ch=3):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            
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
            nn.Conv2d(512, 1, 4, 2, 1, bias=False),
            
            # state size: 1 x 8 x 8
            nn.AvgPool2d(4, 2, 1),
            
            # state size: 1 x 4 x 4
            nn.AvgPool2d(4, 1, 0),
            
            # state size: 1 x 1 x 1
            nn.Sigmoid(),
            )
        
    def forward(self, input):
        return self.model(input)




## TESTING CODE :
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#dis = Discriminator(3).to(device)
#
#gen = Generator(3,3).to(device)
#
#z = torch.randn(1,3,256,256).to(device)
#
#print('Random image size {}'.format(z.size()))
#
#fake = gen(z)
#
#print('Fake image size {}'.format(fake.size()))
#
#print(dis(fake))



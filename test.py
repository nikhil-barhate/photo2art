import os
import torch
from models import Generator
from torchvision.utils import save_image
import torchvision.transforms as transforms
from dataset import ImageDataset
from PIL import Image

batch_size = 1
start_epoch = 4
start_epoch_part = 5

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

dataset = ImageDataset('./data/monet/', transform = transform, train=False)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                         shuffle=True)


device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')


G_monet = Generator().to(device)
G_photo = Generator().to(device)

G_monet.load_state_dict(torch.load('./checkpoints/{}/G_B2A_{}_{}.pth'.format(start_epoch, start_epoch, start_epoch_part)))
G_photo.load_state_dict(torch.load('./checkpoints/{}/G_A2B_{}_{}.pth'.format(start_epoch, start_epoch, start_epoch_part)))

G_monet.eval()
G_photo.eval()

if not os.path.exists('./output/'):
    os.makedirs('./output/')

for i, data in enumerate(dataloader):
    
    image_A, image_B = data
    
    image_A, image_B = image_A.to(device), image_B.to(device)
    
    save_image(image_A,'./output/painting_orig.jpg')
    save_image(image_B,'./output/photo_orig.jpg')
    
    new_A = G_monet(image_B)
    
    new_B = G_photo(image_A)
    
    save_image(new_A,'./output/painting.jpg')
    save_image(new_B,'./output/photo.jpg')
    
    break

#photo = Image.open('./data/monet/testB/2014-08-01 17:41:55.jpg')
#photo.show()
#
#transform = transforms.Compose(transforms.ToTensor())
#
#photo = transform([photo])
#
#painting = G_monet(photo)
#
#
#save_image(painting)





import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImageDataset(Dataset):
   
    def __init__(self, root, transform=None, train=True):
        
        self.transform = transform
        self.root = root
        self.train = train
        
        if self.train:
            self.images_A = sorted(glob(os.path.join(self.root, 'trainA', '*')))
            self.images_B = sorted(glob(os.path.join(self.root, 'trainB', '*')))
        
        else:
            self.images_A = sorted(glob(os.path.join(self.root, 'testA', '*')))
            self.images_B = sorted(glob(os.path.join(self.root, 'testB', '*')))


    def __getitem__(self, index):
        
        if self.transform is not None:    
            image_A = self.transform(Image.open(self.images_A[index % len(self.images_A)]))
            image_B = self.transform(Image.open(self.images_B[index % len(self.images_B)]))
        
        else:
            image_A = Image.open(self.images_A[index % len(self.images_A)])
            image_B = Image.open(self.images_B[index % len(self.images_B)])
        
        return image_A, image_B

    def __len__(self):
        
        # Right method:
        #return max(len(self.images_A), len(self.images_B))
        
        # But to reduce the size of an epoch:
        return 3000



## TESTING CODE :
#transform = transforms.Compose([transforms.ToTensor(),
#            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
#data = ImageDataset('./data/monet/')
#
#print(len(data))
#
#imageA, imageB = data.__getitem__(0)
#
#
#imageA.show()
#imageB.show()


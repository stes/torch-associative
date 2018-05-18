from torchvision import datasets, transforms

import h5py
import torch
from torch import nn

import torch.nn.functional as F

import torch.utils.data

class JointDataset(torch.utils.data.Dataset):
    
    def __init__(self, *datasets):
        
        self.datasets = datasets
    
    def __len__(self):
        
        return min([len(d) for d in self.datasets])
        
    def __getitem__(self, index):
        
        return [ds[index] for ds in self.datasets]
        

def load_dataset(path):
    batch_size_s = 100
    batch_size_t = 1000

    img_size = 32
       #### Noisy Dataset ####

    transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ColorJitter(.1, 1, .75, 0),    
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2881,0.2881,0.2881)),
            transforms.Lambda(lambda x : x.expand([3,-1,-1]))
    ])
    train_mnist = torch.utils.data.DataLoader(
        datasets.MNIST(path, train=True, download=True, transform=transform),
        batch_size=batch_size_t, shuffle=True, num_workers=4)

       #### Clean Dataset ####

    transform = transforms.Compose([
            transforms.Resize(img_size),   
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.188508,    0.19058265,  0.18615675))
    ])
    train_svhn = torch.utils.data.DataLoader(
        datasets.SVHN(path, split='train', download=True, transform=transform),
        batch_size=batch_size_s, shuffle=True, num_workers=4)
    
    return train_svhn, train_mnist
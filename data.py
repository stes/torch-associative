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
        

def load_dataset(path, train=True):
    img_size = 32

    transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ColorJitter(.1, 1, .75, 0),    
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2881,0.2881,0.2881)),
            transforms.Lambda(lambda x : x.expand([3,-1,-1]))
    ])
    mnist = datasets.MNIST(path, train=train, download=True, transform=transform)
    
    transform = transforms.Compose([
            transforms.Resize(img_size),   
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.188508,    0.19058265,  0.18615675))
    ])
    svhn = datasets.SVHN(path, split='train' if train else 'test', download=True, transform=transform)
    
    return {'mnist' : mnist, 'svhn' : svhn}


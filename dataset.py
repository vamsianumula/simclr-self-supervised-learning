from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import random
import torch

class CIFAR10Pair(CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            x_i = self.transform(img)
            x_j = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return x_i, x_j, target

def transform(train=True):
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
    
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

def get_data(batch_size):
    train_data = CIFAR10Pair(root='data', train=True, transform=transform(), download=True)
    val_data = CIFAR10Pair(root='data', train=True, transform=transform(False), download=True)
    test_data = CIFAR10Pair(root='data', train=False, transform=transform(False), download=True)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    
    return train_loader, val_loader, test_loader
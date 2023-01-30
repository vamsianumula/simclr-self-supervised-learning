from PIL import Image
import torchvision
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import get_data
from simclr import simCLR

config = {
    "lr": 1e-3,
    "temperature": 1,
    "batch_size": 64,
    "epochs": 1,
    'num_features':128
}

answers = torch.load('simclr_sanity_check.key')
simclr = simCLR(config)
train_data, val_data, test_data = get_data(config["batch_size"])
simclr.train(train_data)

def err(x,y):
    x = np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
    print(f"Maximum error in data augmentation:{x:.9} ")
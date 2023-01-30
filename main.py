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
    "temperature": 0.5,
    "batch_size": 64,
    "epochs": 1,
    'num_features':128
}

simclr = simCLR(config)
simclr.load_model('trained_simclr.pth')
train_data, val_data, test_data = get_data(config["batch_size"])
simclr.train(train_data)
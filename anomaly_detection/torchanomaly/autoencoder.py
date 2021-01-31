from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from torchvision import datasets
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random


def main():
    data_dir = "/home/jonathan/code/bitsandbytes/anomaly_detection/kerasanomaly/lfw/all"
    epochs = 100
    lr = 1e-3
    test_train = 0.1
    batch = 128
    transform = T.Compose([T.Resize(250), T.CenterCrop(224), T.ToTensor()])  # should
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    train_loader, test_loader = load_data(dataset, test_train, batch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AE()
    model.to(device)
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=w_d)

    model.train()
    start = time.time()
    for epoch in range(epochs):
        ep_start = time.time()
        running_loss = 0.0
        for bx, (data) in enumerate(train_):
            sample = model(data.to(device))
            loss = criterion(data.to(device), sample)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=24, out_channels=12, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        encode = self.enc(x)
        decode = self.dec(encode)
        return decode


def load_data(dataset, test_split, batch_size):
    # Create indices for the split
    dataset_size = len(dataset)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset.dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset.dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

"""
Should I scale, normalize it? -0.5 0.5
"""

from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from torchvision import datasets
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from collections import defaultdict
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import timedelta


def main():
    data_dir = "/home/jonathan/code/bitsandbytes/anomaly_detection/kerasanomaly/lfw/all"
    epochs = 10
    lr = 1e-2  # learning rate
    w_d = 1e-5  # weight decay
    test_train = 0.1
    batch = 1
    transform = T.Compose([T.CenterCrop(224), T.Resize(32), T.ToTensor()])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    train_loader, test_loader = load_data(dataset, test_train, batch)

    metrics = defaultdict(list)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {device}")
    model = AE()
    model.to(device)
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=w_d)

    model.train()
    start = time.time()
    for epoch in range(epochs):
        ep_start = time.time()
        running_loss = 0.0
        for label, data in enumerate(train_loader):
            sample = model(data[0].to(device))
            loss = criterion(data[0].to(device), sample)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(dataset)
        metrics["train_loss"].append(epoch_loss)
        ep_end = time.time()
        print("-----------------------------------------------")
        print("[EPOCH] {}/{}\n[LOSS] {}".format(epoch + 1, epochs, epoch_loss))
        print("Epoch Complete in {}".format(timedelta(seconds=ep_end - ep_start)))
    end = time.time()
    print("-----------------------------------------------")
    print("[System Complete: {}]".format(timedelta(seconds=end - start)))

    _, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.set_title("Loss")
    ax.plot(metrics["train_loss"])
    plt.show()
    return model


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(3072, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.ReLU(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.ReLU(),
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3072),
            nn.ReLU(),
        )

    def forward(self, x):
        orig_shape = x.size()
        x = x.flatten()
        encode = self.enc(x)
        decode = self.dec(encode)
        decode = decode.reshape(orig_shape)
        return decode


# class ConvAE(nn.Module):
#     def __init__(self):
#         super(ConvAE, self).__init__()
#         self.enc = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Linear(784, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 16),
#             nn.ReLU(),
#         )
#         self.dec = nn.Sequential(
#             nn.Linear(16, 32),
#             nn.ReLU(),
#             nn.Linear(32, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#             nn.ReLU(),
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Linear(256, 512),
#             nn.ReLU(),
#             nn.Linear(512, 784),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels=24, out_channels=12, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#         )

#     def forward(self, x):
#         encode = self.enc(x)
#         decode = self.dec(encode)
#         return decode


def load_data(dataset, test_split, batch_size):
    # Create indices for the split
    dataset_size = len(dataset)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset.dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset.dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


if __name__ == "__main__":
    main()
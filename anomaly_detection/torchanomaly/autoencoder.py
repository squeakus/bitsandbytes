"""
Should I scale, normalize it? -0.5 0.5
Why save to state dict?
BCELoss
how do I pass variables into module
relu order
conv mult
"""

from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from torchvision import datasets
from networks import ConvAE, ConvLinearAE, StridedConvAE, StridedAutoEncoder
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import make_grid
from torchvision.utils import save_image
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
from datetime import timedelta


def main(args):
    data_dir = "/home/jonathan/data/lfw"
    action = args[1]
    model_name = args[2]
    imsize = 64
    model = StridedAutoEncoder((3, imsize, imsize))
    epochs = 5
    lr = 1e-2  # learning rate
    w_d = 1e-5  # weight decay
    test_train = 0.1
    batch = 32
    torch.manual_seed(42)

    # set up data
    transform = T.Compose([T.CenterCrop(224), T.Resize(imsize), T.ToTensor()])
    train_loader, test_loader, train_size, test_size = load_data(data_dir, transform, test_train, batch)

    if action == "train":
        model = train(model, epochs, lr, w_d, train_loader, train_size)
        torch.save(model, model_name)
    elif action == "test":
        test(model_name, test_loader)
    elif action == "classify":
        classify(model_name, args[3], transform)

    else:
        print("unrecognised action, use either train, test or classify")
        exit()


def train(model, epochs, lr, w_d, dataloader, datasize):
    metrics = defaultdict(list)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {device}")
    model.to(device)
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=w_d)

    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {total_params}")
    model.train()
    start = time.time()
    for epoch in range(epochs):
        ep_start = time.time()
        running_loss = 0.0
        for image, label in dataloader:
            sample = model(image.to(device))
            loss = criterion(image.to(device), sample)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / datasize
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


def classify(model_name, image_name, transform):
    model = torch.load(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = image_loader(image_name, transform)

    model.eval()
    with torch.no_grad():
        encode = model.enc(image)
        decode = model.dec(encode)
    image = torch.squeeze(image)
    decode = torch.squeeze(decode)

    visualize(f"classified.png", "batch", image.cpu().detach(), decode)


def test(model_name, dataloader):
    model = torch.load(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for batch, label in dataloader:
        model.eval()

        with torch.no_grad():
            encode = model.enc(batch.to(device))
            decode = model.dec(encode)

        for idx, image in enumerate(batch):
            visualize(f"test{idx:02d}.png", "batch", image, decode[idx])

        exit()


def image_loader(image_name, transform):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = transform(image).float()
    image = Variable(image, requires_grad=True)

    image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
    return image.cuda()  # assumes that you're using GPU


def visualize(image_name, title, original, decode):
    """Draws original, encoded and decoded images"""

    image = original.numpy().transpose(1, 2, 0)
    decode_image = decode.cpu().numpy().transpose(1, 2, 0)

    plt.suptitle(title)
    plt.subplot(1, 3, 1)
    plt.title("Original")
    show_image(image)

    # plt.subplot(1, 3, 2)
    # plt.title("Code")
    # plt.imshow(code.reshape([code.shape[-1] // self.rescale, -1]))

    # loss = int(np.sum(np.absolute(img - reco)))
    plt.subplot(1, 3, 3)
    plt.title(f"Decode")
    show_image(decode_image)

    plt.savefig(image_name)
    plt.clf()


def show_image(x):
    plt.imshow(np.clip(x, 0, 1))


def load_data(data_dir, transform, test_split, batch_size):
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    # Create indices for the split
    dataset_size = len(dataset)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset.dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset.dataset, batch_size=batch_size, shuffle=True)

    return (train_loader, test_loader, train_size, test_size)


if __name__ == "__main__":
    if not len(sys.argv) > 2:
        print(
            f"{len(sys.argv)} usage: python autoencoder.py test/train <modelname> or python autoencoder.py classify <modelname> <image_name>"
        )
        exit()
    main(sys.argv)
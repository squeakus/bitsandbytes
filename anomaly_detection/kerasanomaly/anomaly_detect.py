import cv2
import os
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from keras.models import Sequential, Model
import pandas as pd
import tarfile
from math import sqrt
from pathlib import Path


def main():
    if not os.path.exists("out"):
        os.makedirs("out")

    anomaly = Autoencoder("test1", 2000, 75, 75)
    anomaly.train("lfw", 30)


class Autoencoder:
    def __init__(self, savename, codesize, dimx, dimy):
        self.savename = savename
        self.codesize = codesize

        # find a good rescale value for the code block
        divisors = []
        for i in range(1, codesize):
            if codesize % i == 0:
                divisors.append(i)
        self.rescale = divisors[int(len(divisors) / 2)]

        self.dimx = dimx
        self.dimy = dimy

    def train(self, dataset, epochs):
        X = self.load_dataset(dataset)
        X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)
        img_shape = X.shape[1:]
        self.build_autoencoder(img_shape)
        inp = Input(img_shape)
        code = self.encoder(inp)
        reconstruction = self.decoder(code)
        self.autoencoder = Model(inp, reconstruction)
        self.autoencoder.compile(optimizer="adamax", loss="mse")
        print(self.autoencoder.summary())

        for i in range(epochs):
            history = self.autoencoder.fit(x=X_train, y=X_train, epochs=1, validation_data=(X_test, X_test))
            jon = self.load_image("jon.png", True)
            imagename = f"out/jon{i:02}.png"
            self.visualize(imagename, f"Iteration: {i}", jon)

            cat = self.load_image("cat.jpg", True)
            imagename = f"out/cat{i:02}.png"
            self.visualize(imagename, f"Iteration: {i}", cat)

        self.autoencoder.save(self.savename)

    def load_image(self, filename, rescale=False):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.dimx, self.dimy))

        if rescale:
            img = img.astype("float32") / 255.0 - 0.5
        return img

    def load_dataset(self, folder):
        images = []
        for path in Path(folder).rglob("*.jpg"):
            images.append(self.load_image(str(path)))

        images = np.stack(images).astype("uint8")
        images = images.astype("float32") / 255.0 - 0.5
        return images

    def visualize(self, imagename, title, img):
        """Draws original, encoded and decoded images"""
        # img[None] will have the same shape as the model input
        code = self.encoder.predict(img[None])[0]
        reco = self.decoder.predict(code[None])[0]

        plt.suptitle(title)
        plt.subplot(1, 3, 1)
        plt.title("Original")
        show_image(img)

        plt.subplot(1, 3, 2)
        plt.title("Code")
        plt.imshow(code.reshape([code.shape[-1] // self.rescale, -1]))

        plt.subplot(1, 3, 3)
        plt.title("Reconstructed")
        show_image(reco)

        plt.savefig(imagename)
        plt.clf()

    def build_autoencoder(self, img_shape):
        # The encoder
        self.encoder = Sequential()
        self.encoder.add(InputLayer(img_shape))
        self.encoder.add(Flatten())
        self.encoder.add(Dense(self.codesize))

        # The decoder
        self.decoder = Sequential()
        self.decoder.add(InputLayer((self.codesize,)))
        self.decoder.add(
            Dense(np.prod(img_shape))
        )  # np.prod(img_shape) is the same as 32*32*3, it's more generic than saying 3072
        self.decoder.add(Reshape(img_shape))


def show_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1))


if __name__ == "__main__":
    main()
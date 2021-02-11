import torch.nn as nn


class ConvLinearAE(nn.Module):
    def __init__(self, size):
        super(ConvLinearAE, self).__init__()
        self.size = size
        self.flatsize = size[0] * size[1] * size[2]

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.Flatten(),
            nn.Linear(98304, 1024),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(1024, 98304),
            nn.ReLU(),
            nn.Unflatten(1, (24, 64, 64)),
            # nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.ConvTranspose2d(in_channels=24, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        encode = self.enc(x)
        print(encode.shape)
        decode = self.dec(encode)
        return decode


class StridedAutoEncoder(nn.Module):
    def __init__(self, size):
        super(StridedAutoEncoder, self).__init__()
        self.size = size
        self.flatsize = size[0] * size[1] * size[2]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=5, stride=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return decode


class StridedConvAE(nn.Module):
    def __init__(self, size):
        super(StridedConvAE, self).__init__()
        self.size = size
        self.flatsize = size[0] * size[1] * size[2]

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        encode = self.enc(x)
        decode = self.dec(encode)
        return decode


class ConvAE(nn.Module):
    def __init__(self, size):
        super(ConvAE, self).__init__()
        self.size = size
        self.flatsize = size[0] * size[1] * size[2]

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=24, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        encode = self.enc(x)
        print(encode.shape)
        decode = self.dec(encode)
        return decode


class SimpleAE(nn.Module):
    def __init__(self, size):
        super(AE, self).__init__()
        self.size = size
        self.flatsize = size[0] * size[1] * size[2]

        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(self.flatsize, 5000), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(5000, self.flatsize), nn.ReLU(), nn.Unflatten(1, self.size))

    def forward(self, x):
        orig_shape = x.size()
        encode = self.enc(x)
        decode = self.dec(encode)
        # decode = decode.reshape(orig_shape)
        return decode


class LinearAE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.enc = nn.Sequential(
            #        orig_shape = x.size()
            nn.Flatten(),
            nn.Linear(3072, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.ReLU(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.ReLU(),
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3072),
            nn.ReLU(),
            nn.Unflatten(1, (3, 32, 32)),
        )


def forward(self, x):
    orig_shape = x.size()
    encode = self.enc(x)
    decode = self.dec(encode)
    # decode = decode.reshape(orig_shape)
    return decode


# https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
# class ConvEncoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         self.encoder = nn.Sequential( # like the Composition layer you built
#             nn.Conv2d(1, 16, 3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, 3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 7)
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(64, 32, 7),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
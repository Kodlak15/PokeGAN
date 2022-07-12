import torch
import torch.nn as nn
import os
from os.path import join
from utils import get_paths

root_dir, img_dir, train_dir, fake_dir, weights_dir, history_dir = get_paths()

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        latent_size = 128

        self.net = nn.Sequential(
            # in: 3 x 128 x 128

            nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 128 x 64 x 64

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 256 x 32 x 32

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 512 x 16 x 16

            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 1024 x 8 x 8

            nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 2048 x 4 x 4

            nn.Conv2d(2048, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # out: 1 x 1 x 1

            nn.Flatten(),
            nn.Sigmoid()
        )

        if "Discriminator.pth" in os.listdir(weights_dir):
            self.load()

    def forward(self, x):
        out = self.net(x)
        return out

    def save(self):
        torch.save(self.state_dict(), join(weights_dir, "Discriminator.pth"))

    def load(self):
        assert "Discriminator.pth" in os.listdir(weights_dir), "No discriminator weights found"
        self.load_state_dict(torch.load(join(weights_dir, "Discriminator.pth")))
        print("Discriminator weights loaded successfully!")

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        latent_size = 128

        self.net = nn.Sequential(
            # in: latent_size x 1 x 1

            nn.ConvTranspose2d(latent_size, 2048, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(True),
            # out: 2048 x 4 x 4

            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # out: 1024 x 8 x 8

            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # out: 512 x 16 x 16

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # out: 256 x 32 x 32

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # out: 128 x 64 x 64

            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # out: 3 x 128 x 128
        )

        if "Generator.pth" in os.listdir(weights_dir):
            self.load()

    def forward(self, x):
        out = self.net(x)
        return out

    def save(self):
        torch.save(self.state_dict(), join(weights_dir, "Generator.pth"))

    def load(self):
        assert "Generator.pth" in os.listdir(weights_dir), "No generator weights found"
        self.load_state_dict(torch.load(join(weights_dir, "Generator.pth")))
        print("Generator weights loaded successfully!")
from typing import Tuple, Union
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from PIL import Image
import os
from os.path import join
import cv2
from device_mgmt import to_device
import json
import numpy as np
import re

def format_path(path: str):
    return path.replace("\\", '/')

def prepare_paths():
    root_dir = os.getcwd()
    img_dir = format_path(join(root_dir, "images"))
    train_dir = format_path(join(img_dir, "train-images"))
    fake_dir = join(root_dir, "fakes")
    weights_dir = join(root_dir, "weights")
    history_dir = join(root_dir, "history")

    with open("paths.txt", 'w') as f:
        f.write(root_dir + '\n')
        f.write(img_dir + '\n')
        f.write(train_dir + '\n')
        f.write(fake_dir + '\n')
        f.write(weights_dir + '\n')
        f.write(history_dir)

    if not "images" in os.listdir(root_dir):
        os.mkdir(img_dir)

    if not "train-images" in os.listdir(img_dir):
        os.mkdir(join(img_dir, "train-images"))

    if not "fakes" in os.listdir(root_dir):
        os.mkdir(fake_dir)

    if not "weights" in os.listdir(root_dir):
        os.mkdir(weights_dir)

    if not "history" in os.listdir(root_dir):
        os.mkdir(history_dir)
        with open(join(history_dir, "history.json"), 'w') as f:
            json.dump({"history": []}, f)

def get_paths():
    prepare_paths()
    with open("paths.txt", 'r') as f:
        return [path.replace('\n', '') for path in f.readlines()]

root_dir, img_dir, train_dir, fake_dir, weights_dir, history_dir = get_paths()
train_stats = [0.1874, 0.1779, 0.1681], [1.0, 1.0, 1.0]

@torch.no_grad()
def show_images(batch: Union[DataLoader, Tensor]):
    """
    Takes a tensor (B, C, W, H) or dataloader as input and displays a batch of training images
    """
    for images in batch:
        fig, ax = plt.subplots(figsize=(32,32))
        ax.set_xticks([]); ax.set_yticks([])
        images = images.to("cpu")
        images = denormalize(images, *train_stats)
        ax.imshow(make_grid(images, nrow=8).permute(1, 2, 0))
        break

@torch.no_grad()
def show_fakes(images, num_to_show=64):
    """
    Displays a collection of fake images
    """
    fig, ax = plt.subplots(figsize=(32,32))
    ax.set_xticks([]); ax.set_yticks([])
    images = images.to("cpu")
    images = denormalize(images[:num_to_show], *train_stats)
    ax.imshow(make_grid(images, nrow=8).permute(1, 2, 0))


def transform_image(img: Image):
    """
    Intended to transform images from CMYK -> RGB
    Overlays image on a plain, black background to remove transparent pixels
    """
    new_img = Image.new("RGBA", img.size, "BLACK")
    new_img.paste(img, (0, 0), img)
    new_img = new_img.convert("RGB")
    return new_img

def save_samples(G: nn.Module, index: int, x: Tensor):
    fake_images = denormalize(to_device(G(x), device="cpu"), *train_stats)
    filename = join(fake_dir, "generated-images-{0:0=4d}.png".format(index))
    print(f"Saving {filename}")
    save_image(fake_images[:64], filename, nrow=8)

def make_video(fps=30):
    vid_fname = "pokeGAN.avi"

    files = [join(fake_dir, f) for f in os.listdir(fake_dir) if 'generated' in f]
    files.sort()

    out = cv2.VideoWriter(vid_fname, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (1042, 1042))
    [out.write(cv2.imread(fname)) for fname in files]
    out.release()

def denormalize(images, means, stds):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means

def parse_history():
    with open(join(os.getcwd(), "history", "history.json"), 'r') as f:
        history = json.load(f)["history"]

    epochs = np.arange(len(history))
    losses_d = []
    losses_g = []
    real_scores = []
    fake_scores = []

    for epoch in history:
        loss_g, loss_d, real_score, fake_score = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", epoch)
        losses_g.append(float(loss_g))
        losses_d.append(float(loss_d))
        real_scores.append(float(real_score))
        fake_scores.append(float(fake_score))

    return epochs, losses_d, losses_g, real_scores, fake_scores

def plot_results():
    epochs, losses_d, losses_g, real_scores, fake_scores = parse_history()

    plt.style.use("seaborn")
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 4))
    ax[0][0].title.set_text("Losses")
    ax[0][1].title.set_text("Scores")
    # Plot losses
    ax[0][0].plot(epochs, losses_d)
    ax[0][0].plot(epochs, np.poly1d(np.polyfit(epochs, losses_d, deg=2))(epochs), color="red", linestyle="--")
    ax[0][0].set_ylim([-0.2, 5])
    ax[0][0].set_ylabel("Discriminator")
    ax[1][0].plot(epochs, losses_g)
    ax[1][0].plot(epochs, np.poly1d(np.polyfit(epochs, losses_g, deg=2))(epochs), color="red", linestyle="--")
    ax[1][0].set_ylim([-0.2, 5])
    ax[1][0].set_ylabel("Generator")
    # Plot scores
    ax[0][1].plot(epochs, real_scores)
    ax[0][1].plot(epochs, np.poly1d(np.polyfit(epochs, real_scores, deg=2))(epochs), color="red", linestyle="--")
    ax[0][1].set_ylabel("Real")
    ax[1][1].plot(epochs, fake_scores)
    ax[1][1].plot(epochs, np.poly1d(np.polyfit(epochs, fake_scores, deg=2))(epochs), color="red", linestyle="--")
    ax[1][1].set_ylabel("Fake")

    plt.show()
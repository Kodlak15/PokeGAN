from turtle import bgcolor
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

train_stats = [0.1874, 0.1779, 0.1681], [1.0, 1.0, 1.0]

root_dir = os.getcwd()
fake_dir = join(root_dir, "fakes")
weights_dir = join(root_dir, "weights")

if not "fakes" in os.listdir(root_dir):
    os.mkdir(fake_dir)

if not "weights" in os.listdir(root_dir):
    os.mkdir(weights_dir)

def format_path(path: str):
    return path.replace("\\", '/')

def denormalize(images: Tensor, stats: Tuple[Tuple[int]]):
    return (images * stats[1][0]) + stats[0][0]

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

def transform_image(img: Image):
    """
    Intended to transform images from CMYK -> RGB
    Overlays image on a plain, white background to remove transparent pixels
    """
    new_img = Image.new("RGBA", img.size, "BLACK")
    new_img.paste(img, (0, 0), img)
    new_img = new_img.convert("RGB")
    return new_img

def save_samples(G: nn.Module, index: int, x: Tensor):
    fake_images = denormalize(to_device(G(x), device="cpu"), *train_stats)
    #fake_images = G(x)
    filename = join(fake_dir, "generated-images-{0:0=4d}.png".format(index))
    print(f"Saving {filename}")
    save_image(fake_images, filename, nrow=8)

def make_video(fps=10):
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
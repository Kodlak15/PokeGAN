import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from device_mgmt import *
from models import Discriminator, Generator
from utils import make_video, save_samples
import os
from os.path import join
import json
import shutil

root_dir = os.getcwd()
history_dir = join(root_dir, "history")
weights_dir = join(root_dir, "weights")

if not "history" in os.listdir(root_dir):
    os.mkdir(history_dir)
    with open(join(history_dir, "history.json"), 'w') as f:
        json.dump({"history": []}, f)

latent_size = 128

def train_discriminator(D: nn.Module, G: nn.Module, images: Tensor, opt_d: torch.nn.functional, device: torch.device):
    opt_d.zero_grad()

    real_preds = D(images)
    real_targets = torch.ones(images.size(0), 1, device=device) # Make noisy
    real_noisy_targets = (0.7 - 1.2) * torch.rand(images.size(0), 1, device=device) + 1.2
    real_loss = F.binary_cross_entropy(real_preds, real_noisy_targets)
    real_score = torch.mean(real_preds).item()

    x = torch.randn(images.size(0), latent_size, 1, 1, device=device)
    fake_images = G(x)

    fake_preds = D(fake_images)
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device) # Make noisy
    fake_noisy_targets = (0.0 - 0.3) * torch.rand(fake_images.size(0), 1, device=device) + 0.3
    fake_loss = F.binary_cross_entropy(fake_preds, fake_noisy_targets)
    fake_score = torch.mean(fake_preds).item()

    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()

    return loss.item(), real_score, fake_score

def train_generator(D: nn.Module, G: nn.Module, batch_size: int, opt_g: torch.nn.functional, device: torch.device):
    opt_g.zero_grad()

    x = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = G(x)

    preds = D(fake_images)
    targets = torch.ones(batch_size, 1, device=device)
    loss = F.binary_cross_entropy(preds, targets)

    loss.backward()
    opt_g.step()
    
    return loss.item()

def fit(D: nn.Module, G: nn.Module, train_dl: DataLoader, epochs: int, lr: float, device: torch.device, start_idx=1):
    clear_cache_and_get_info(device)

    # Create backups in case model collapses during training
    shutil.copy(join(history_dir, "history.json"), join(history_dir, "history_backup.json"))

    if "Discriminator.pth" in os.listdir(weights_dir):
        shutil.copy(join(weights_dir, "Discriminator.pth"), join(weights_dir, "Discriminator-backup.pth"))

    if "Generator.pth" in os.listdir(weights_dir):
        shutil.copy(join(weights_dir, "Generator.pth"), join(weights_dir, "Generator-backup.pth"))

    with open(join(history_dir, "history.json"), 'r') as f:
        history = json.load(f)["history"]

    losses_d = []
    losses_g = []
    real_scores = []
    fake_scores = []

    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for real_images in tqdm(train_dl):
            loss_d, real_score, fake_score = train_discriminator(D, G, real_images, opt_d, device)
            loss_g = train_generator(D, G, real_images.size(0), opt_g, device)

        losses_d.append(loss_d)
        losses_g.append(loss_g)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        epoch_results = "Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
                        epoch+1, epochs, loss_g, loss_d, real_score, fake_score)
        
        print(epoch_results)
        history.append(epoch_results[epoch_results.find(',')+2:])

        x = torch.randn(64, latent_size, 1, 1, device=device)
        save_samples(G, epoch+start_idx, x)
        D.save()
        G.save()

    print("Saving results...")
    with open(join(history_dir, "history.json"), 'w') as f:
        json.dump({"history": history}, f)

    make_video()

    return history
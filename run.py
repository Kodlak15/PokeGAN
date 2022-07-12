from random import shuffle
import torchvision.transforms as T
from torch.utils.data import DataLoader
from utils import transform_image, format_path, get_paths
from device_mgmt import get_default_device, DeviceDataLoader, to_device
from models import Discriminator, Generator
from dataset import PokemonDataset
from train import fit
import os
from os.path import join

def run():
    try:
        epochs = int(input("Enter the number of epochs to train for (1-5000): "))
        assert epochs in range(1, 5001), "Enter a number between 1 and 5000"
        lr = float(input("Enter the learning rate (5e-6 - 5e-4): "))
        assert lr >= 5e-6 and lr <= 5e-4, "Enter a number between 5e-6 and 5e-4"

    except AssertionError:
        print("Restarting program...")
        run()

    device = get_default_device()
    train_ds = PokemonDataset(train_dir, transform=train_transform)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    train_dl = DeviceDataLoader(train_dl, device)
    D = to_device(Discriminator(), device)
    G = to_device(Generator(), device)

    return fit(D, G, train_dl, epochs, lr, device, start_idx=len(os.listdir(fake_dir))+1)

if __name__ == "__main__":
    root_dir, img_dir, train_dir, fake_dir, weights_dir, history_dir = get_paths()
    
    img_size = 128
    batch_size = 256
    train_stats = [0.1874, 0.1779, 0.1681], [1.0, 1.0, 1.0]

    train_transform = T.Compose([
        T.Lambda(lambda img: transform_image(img)),
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.RandomHorizontalFlip(0.2),
        T.RandomRotation(3, fill=0),
        T.ToTensor(),
        T.Normalize(*train_stats)
    ])

    run()
    input("Training finished, press enter to exit...")
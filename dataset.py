from utils import format_path
from torch.utils.data import Dataset
from PIL import Image
import os
from os.path import join

class PokemonDataset(Dataset):
    """ Pokemon images dataset """
    def __init__(self, directory: str, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_paths = [format_path(join(directory, img_name)) for img_name in sorted(os.listdir(directory))]

    def __len__(self):
        return len([f for f in os.listdir(self.directory) if ".png" in f])

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        with Image.open(path) as img:

            if self.transform:
                img = self.transform(img)
            
            return img
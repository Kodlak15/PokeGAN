import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
from PIL import Image
import os
from os.path import join
from tqdm import tqdm
import random
import json
from utils import format_path

root_dir = os.getcwd()
img_dir = join(root_dir, "images")
train_dir = format_path(join(img_dir, "train-images"))

if not "images" in os.listdir(root_dir):
    os.mkdir(img_dir)

def get_image_urls(refresh=False, seed=15):
    url = "https://www.pokemon.com/us/pokedex/"
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")
    body = soup.find("body")
    pattern = "\/us\/pokedex\/[a-zA-z]+"
    pokemon = body.find_all('a', {"href": re.compile(pattern)})
    
    if refresh or "img-urls.json" not in os.listdir(img_dir):
        img_urls = []
        for p in tqdm(pokemon):
            p_url = urljoin(url, p["href"])
            r = requests.get(p_url)
            soup = BeautifulSoup(r.content, "html.parser")
            body = soup.find("body")
            img_url = body.find("img", {"class": "active"})["src"]
            img_urls.append(img_url)

        print(f"{len(img_urls)} Pokemon images found.")
        random.seed(seed)
        random.shuffle(img_urls)

        filename = join(img_dir, "img-urls.json")
        with open(filename, 'w') as f:
            json.dump(img_urls, f)

    filename = join(img_dir, "img-urls.json")
    with open(filename, 'r') as f:
        img_urls = json.load(f)

    return img_urls

def get_images(refresh=False, seed=15):
    if not "img-urls.json" in os.listdir(img_dir):
        img_urls = get_image_urls(refresh=True, seed=seed)

    else:
        img_urls = get_image_urls(seed=seed)

    if not "train-images" in os.listdir(img_dir):
        os.mkdir(join(img_dir, "train-images"))

    if len(os.listdir(train_dir)) == 0:
        for url in tqdm(img_urls):
            "Creating training set..."
            pID = url.split('/')[-1]
            r = requests.get(url)
            if r.status_code == 200:
                r.raw.decode_content = True

                if pID not in os.listdir(join(img_dir, "train-images")):
                    filename = join(img_dir, "train-images", pID)
                    with open(filename, 'wb') as f:
                        f.write(r.content)

            else:
                print(f"Image {pID} could not be retrieved")

    print("Finished!")

def main(refresh=False, seed=15):
    """
    Gets all Pokemon images from https://www.pokemon.com/us/pokedex/
    Images are saved in the images directory under the root directory
    """
    get_images()

if __name__ == "__main__":
    refresh = False
    seed = 15 
    main(refresh=refresh, seed=seed)
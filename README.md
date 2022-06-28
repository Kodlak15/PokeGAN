This project is a work in progress. As such, be advised that bugs may exist and performance may be sub-optimal. It was originally started as a final project for an online deep learing course I completed. The course can be found at this url -> https://jovian.ai/learn/deep-learning-with-pytorch-zero-to-gans. 

For this project I am using the PyTorch framework to build a Generative Adversarial Network (GAN) that takes as input images of all 905 Pokemon in the official Pokedex, and produces new images of fake Pokemon.

A brief overview of the files in this repository:
  - PokeScraper.py: The script used to retrieve the images from https://www.pokemon.com/us/pokedex/.
  - dataset.py: Contains a custom PyTorch dataset for the Pokemon images.
  - device_mgmt.py: Contains a series of helper functions and classes that are used to inspect memory issues or facilitate movement of data between devices.
  - models.py: Contains two custom PyTorch models: the generator and the discriminator. The discriminator is a convolutional neural network whose goal is to correctly predict whether images are real or fake, while the generator is a deconvolutional neural network whose goal is to create images that fool the discriminator. 
  - train.py: A series of helper functions that facilitate the training process for the two networks. 
  - utils.py: A collection of general helper functions. Utilities for saving predictions, manipulating images, altering file names, etc.
  - run.py: The script that brings everything together and starts the training process. In the command line navigate to the directory this repository has been cloned to, then run "python3 run.py". You will be asked for the number of epochs you would like to train for as well as the desired learning rate. The script will run in the command line and save results after each epoch. 
  - PokeGAN.ipynb: The Jupyter notebook version of this project. I would advise against using this file unless you want to tinker with it, as it is not as well maintained as the rest of the project and is in some ways completely outdated. I intend to clean this up once I am happy with the rest of the project.

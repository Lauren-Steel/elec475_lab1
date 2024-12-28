import torch
import random
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
from model import autoencoderMLP4Layer
from test import test_autoencoder, denoise_image, interpolate

if __name__ == '__main__':

    #   read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-l', metavar='load_file', type=str, help='parameter file (.pth)')  # Changed '-s' to '-l'
    args = argParser.parse_args()

    load_file = None 
    if args.l != None:
        load_file = args.l
    

    batch_size = 32
    bottleneck_size = 8

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)

    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_transform = train_transform

    test_set = MNIST('./data/mnist', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    N_input = 28 * 28   # MNIST image size
    N_output = N_input
    model = autoencoderMLP4Layer(N_input=N_input, N_bottleneck=bottleneck_size, N_output=N_output)
    model.load_state_dict(torch.load(load_file))  # Loads pretrained weights using 'load_file'
    model.to(device)
    

    idx = random.sample(range(len(test_set)), 1)
    idx = idx[0]
    print(f"Index: {idx}")

    # Step 4
    test_autoencoder(model, idx, test_set, device)
    
    # Step 5
    denoise_image(model, idx, test_set, device)

    # Step 6
    interpolate(model, test_loader, device, steps=8)
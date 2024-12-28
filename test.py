
import torch
import random
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
from model import autoencoderMLP4Layer

# 4. Test Autoencoder
def test_autoencoder(model, idx, test_set, device):
    print("Testing Autoencoder")
    
    model.eval()
    
    if 0 <= idx <= test_set.data.size()[0]:
        print('label = ', test_set.targets[idx].item())

        # Retrieves image data 
        img = test_set.data[idx]

        # Converts the image data to torch.float32
        img = img.type(torch.float32)

        # Normalizes the image data to the range [0, 1]
        img = (img - torch.min(img)) / torch.max(img)

        img = img.to(device=device)

        # Reshapes image data to format suitable for the model input: [1, 784] and torch.float32
        img = img.view(1, img.shape[0]*img.shape[1]).type(torch.FloatTensor)

        # Performs inference using the model to get reconstructed output
        # Ensures no gradients are computed during inference - saves memory and computation
        with torch.no_grad():
            output = model(img)

        # Reshapes output to original image dimensions
        output = output.view(28, 28).type(torch.FloatTensor)


        img = img.view(28, 28).type(torch.FloatTensor)

        f = plt.figure()
        f.add_subplot(1,2,1)
        plt.imshow(img, cmap='gray')
        f.add_subplot(1,2,2)
        plt.imshow(output, cmap='gray')
        plt.show()



# 5. Denoising Image
def denoise_image(model, idx, test_set, device, noise_level=0.3):
    print("Denoising Image")

    model.eval()
    if 0 <= idx <= test_set.data.size()[0]:
        # Retrieves image data 
        img = test_set.data[idx]
        img = img.type(torch.float32)
        img = (img - torch.min(img)) / torch.max(img)
        img = img.to(device=device)

        # Reshapes image data to format suitable for the model input: [1, 784] and torch.float32
        img = img.view(1, img.shape[0]*img.shape[1]).type(torch.FloatTensor)

        # Add uniform noise to the image
        noisy_img = img + noise_level * torch.rand(img.size()).to(device)
        noisy_img = torch.clamp(noisy_img, 0.0, 1.0)

        # Pass the noisy image through the autoencoder
        with torch.no_grad():
            output = model(noisy_img.view(1, -1))


        # Reshapes output to original image dimensions
        output = output.view(28, 28).type(torch.FloatTensor)
        noisy_img = noisy_img.view(28, 28).type(torch.FloatTensor)


        img = img.view(28, 28).type(torch.FloatTensor)


        f = plt.figure()
        f.add_subplot(1, 3, 1)  # 3 subplots now: original, noisy, reconstructed
        plt.imshow(img.cpu(), cmap='gray')
        plt.title("Original")
        f.add_subplot(1, 3, 2)
        plt.imshow(noisy_img.cpu(), cmap='gray')
        plt.title("Noisy")
        f.add_subplot(1, 3, 3)
        plt.imshow(output.cpu(), cmap='gray')
        plt.title("Reconstructed")
        plt.show()



            





# 6. Bottleneck Interpolation - linear interpolations function
def interpolate(model, test_loader, device, steps):
    print("Bottleneck Interpolation")
    model.eval()

    # encode images to get their bottleneck representations
    with torch.no_grad():
        fig, axs = plt.subplots(3, steps + 2, figsize=(15, 8))
        shuffled_data = torch.utils.data.DataLoader(test_loader.dataset, batch_size=test_loader.batch_size, shuffle=True)

        for idx, (imgs, _) in enumerate(shuffled_data):
            if idx >= 3:  # To ensure we only plot for three batches to fit our 3xN grid
                break
                     
           
            img1, img2 = imgs[0], imgs[1]
            # reshape images to (1, 784) and send to device
            img1 = img1.view(1, -1).to(device)
            img2 = img2.view(1, -1).to(device) 

            tensor_1 = model.encode(img1)
            tensor_2 = model.encode(img2)

            tensors = []
            # linearly interpolate through the two tensors for n steps -> creates set of n new bottleneck tensors
            for i in torch.linspace(0, 1, steps): # interpolation weights
                interp_tensors = i * tensor_1 + (1 - i) * tensor_2
                # pass each through decode method
                output = model.decode(interp_tensors.unsqueeze(0)).view(1, 28, 28)
                tensors.append(output.cpu().numpy().squeeze()) 

            # Set the first and last image in the batch row
            axs[idx, 0].imshow(img1.cpu().view(28, 28), cmap='gray')
            axs[idx, -1].imshow(img2.cpu().view(28, 28), cmap='gray')
        
            # Plot interpolated images
            for k, tensor in enumerate(tensors):
                axs[idx, k + 1].imshow(tensor, cmap='gray')  # Use `k + 1` because the first column is img1

        plt.show()




if __name__ == '__main__':
    print('running main ...')

    #   read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argParser.add_argument('-z', metavar='bottleneck size', type=int, help='int [32]')
    argParser.add_argument('-b', metavar='batch size', type=int, help='batch size [32]')

    args = argParser.parse_args()

    save_file = None
    if args.s != None:
        save_file = args.s
    bottleneck_size = 0
    if args.z != None:
        bottleneck_size = args.z
    if args.b != None:
        batch_size = args.b

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
    model.load_state_dict(torch.load(save_file)) # loads pretrained weights
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





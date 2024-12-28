import torch
import torch.nn.functional as F
import torch.nn as nn


class autoencoderMLP4Layer(nn.Module):
    
    def __init__(self, N_input=784, N_bottleneck=8, N_output=784):
        super(autoencoderMLP4Layer, self).__init__()
        N2 = 392
        self.fc1 = nn.Linear(N_input, N2)
        self.fc2 = nn.Linear(N2, N_bottleneck)
        self.fc3 = nn.Linear(N_bottleneck, N2)
        self.fc4 = nn.Linear(N2, N_output)
        self.type = 'MLP4'
        self.input_shape = (1, 28*28)

    def encode(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)  
        return x

    def decode(self, x):
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        return self.decode(self.encode(x))




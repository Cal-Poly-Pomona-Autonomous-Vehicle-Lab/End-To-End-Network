import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module): 
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(64, 32, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(32, 16, kernel_size=(3,3))
        self.conv3 = nn.Conv2d(16, 8, kernel_size=(3,3))
        self.conv4 = nn.Conv2d(8, 4, kernel_size=(3,3))
        self.conv5 = nn.Conv2d(4, 1, kernel_size=(3,3))
    
    def foward(self, x): 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = nn.Sigmoid(x)

        return x

class Generator(nn.Module): 
    def __init__(self, noise_size, conv_dim): 
        super(Discriminator, self).__init__() 
        self.deconv1 = nn.ConvTranspose2d(1, 4, kernel_size=(3, 3))
        self.deconv2 = nn.ConvTranspose2d(4, 8, kernel_size=(3, 3))
        self.deconv3 = nn.ConvTranspose2d(16, 32, kernel_size=(3, 3))
        self.deconv4 = nn.ConvTranspose2d(32, 64, kernel_size=(3, 3))

    def forward(self, x): 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        return x
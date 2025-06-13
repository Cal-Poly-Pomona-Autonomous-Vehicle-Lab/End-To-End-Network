import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module): 
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3))
        self.conv4 = nn.Conv2d(128, 1, kernel_size=(3,3))
        self.sig = nn.Sigmoid()
    
    def forward(self, x): 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = self.sig(x)

        return x

class Generator(nn.Module): 
    def __init__(self, noise_size, conv_dim): 
        super(Generator, self).__init__() 
        self.deconv0 = nn.ConvTranspose2d(100, 1024, kernel_size=(2,2), stride=1)
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=(2, 2), stride=2)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=(2, 2), stride=(2,2), padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2,2), padding=1)
        self.deconv4 = nn.ConvTranspose2d(128, 3, kernel_size=(3, 3), stride=(2,2), padding=1)

    def forward(self, x): 
        x = F.relu(self.deconv0(x))
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))

        return x
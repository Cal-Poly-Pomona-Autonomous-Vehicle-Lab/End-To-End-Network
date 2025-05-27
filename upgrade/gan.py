import numpy as np 
import torch 
import torch.nn as nn


# VGG 16 Model 
class Generator(nn.Module): 
    def __init__(self): 
        super(Generator, self).__init__()
        self.conv1x1 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1)
        self.conv1x2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.maxpooling1 = nn.MaxPool2d((2, 2), stride=2)

        self.conv2x1 = nn.Conv2d(64, 128, kernel_size=(2,2), padding=1)
        self.conv2x2 = nn.Conv2d(128, 128, kernel_size=(2,2))
        self.maxpooling2 = nn.MaxPool2d((2,2), stride=2)

        self.conv3x1 = nn.Conv2d(128, 256, (2,2), padding=1)
        self.conv3x2 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv3x3 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.maxpooling3 = nn.MaxPool2d((2,2), stride=2)

        self.conv4x1 = nn.Conv2d(256, 512, (3, 3), padding=1)
        self.conv4x2 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv4x3 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.maxpooling4 = nn.MaxPool2d((2,2), stride=2)

        self.conv5x1 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv5x2 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv5x3 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.maxpooling5 = nn.MaxPool2d((2,2), stride=2)
        
        self.li1 = nn.Linear(25088, 4096)
        self.li2 = nn.Linear(4096, 4096)
        self.li3 = nn.Linear(4096, 1000)

    def forward(self, x): 
        x = self.conv1x1(x)
        x = self.conv1x2(x)
        x = self.maxpooling1(x)

        x = self.conv2x1(x)
        x = self.conv2x2(x)
        x = self.maxpooling2(x)

        x = self.conv3x1(x)
        x = self.conv3x2(x)
        x = self.conv3x3(x)
        x = self.maxpooling3(x)

        x = self.conv4x1(x)
        x = self.conv4x2(x)
        x = self.conv4x3(x)
        x = self.maxpooling4(x)

        x = self.conv5x1(x)
        x = self.conv5x2(x)
        x = self.conv5x3(x)
        x = self.maxpooling5(x)

        x = x.flatten(start_dim=1)

        x = self.li1(x)
        x = self.li2(x)
        x = self.li3(x)

        x = nn.Softmax(x)

        return x


class Discriminator(nn.Module): 
    def __init__(self): 
        super(Discriminator, self).__init__()
    
    def forward(self): 
        pass
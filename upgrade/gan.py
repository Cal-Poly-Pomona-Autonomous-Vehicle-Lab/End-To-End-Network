import numpy as np 
import torch 
import torch.nn as nn


# VGG 16 Model - https://arxiv.org/pdf/1409.1556
class Discriminator(nn.Module): 
    def __init__(self): 
        super(Discriminator, self).__init__()
        pass 

    def forward(self, x): 
        pass

# http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-handout.pdf
class Generator(nn.Module): 
    def __init__(self): 
        super(Generator, self).__init__()

        # Encoding
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1)

        # Redidual Block

        # Decoding
    
    def forward(self, x): 
        x = self.conv1(x)
        print(x.shape)

        return x
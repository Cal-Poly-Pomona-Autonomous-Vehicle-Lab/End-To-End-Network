import torch
import torchvision 

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn

from PIL import Image

# End to End Self Driving Cars 
# https://developer.nvidia.com/blog/deep-learning-self-driving-cars/
# https://arxiv.org/pdf/1604.07316v1

# Car Behavior Cloning
# https://github.com/naokishibuya/car-behavioral-cloning

class End_to_End_NN(nn.Module):
    def __init__(self):
        super(End_to_End_NN, self).__init__()
        self.norm = torch.nn.LayerNorm(normalized_shape=[66, 200])

        self.conv1 = torch.nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2))
        self.conv2 = torch.nn.Conv2d(24, 36, kernel_size=(5, 5), stride=(2,2)) 
        self.conv3 = torch.nn.Conv2d(36, 48, kernel_size=(5,5), stride=(2,2))
        self.conv4 = torch.nn.Conv2d(48, 64, kernel_size=(3,3))   
        self.conv5 = torch.nn.Conv2d(64, 64, kernel_size=(3,3))

        self.li1 = torch.nn.Linear(1153, 100) 
       	self.li2 = torch.nn.Linear(100, 50) 
       	self.li3 = torch.nn.Linear(50, 10) 
       	self.li4 = torch.nn.Linear(10, 1)  

    def forward(self, x, steering):
        x = self.norm(x)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x)) 
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x)) 
    
        x = x.flatten(start_dim=1)
        x = torch.cat((x, steering), dim=1)

        x = self.li1(x)
        x = self.li2(x)
        x = self.li3(x)
        x = self.li4(x)

        return x 

    

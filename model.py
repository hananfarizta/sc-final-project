# initialize model

import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict
import torch.nn.functional as F


class Model(nn.Module):
    def _init_(self, in_channels=1):
        super(Model, self)._init_()
        
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.maxpool2d = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(2304, 128)
        self.drop = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
    
        x = self.conv1(x)
        x = self.maxpool2d(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.maxpool2d(x)
        x = F.relu(x)
            
        x = x.view(x.size(0), -1)
            
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
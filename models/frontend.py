import torch.nn as nn
from .conv3dNet import Conv3dNetwork
import torch.nn.functional as F
from .resnet import Resnet
import config


class Frontend(nn.Module):
    '''This is the frontend class'''

    def __init__(self):
        super(Frontend, self).__init__()
        self.Conv3dNetwork = Conv3dNetwork()
        self.norm = nn.BatchNorm3d(config.frontend["3dCNN"]["outputChannels"])
        self.pool = nn.MaxPool3d(config.frontend["pool"]["kernel"], stride=config.frontend["pool"]["stride"],
                                 padding=config.frontend["pool"]["padding"])
        self.resnet = Resnet()

    def forward(self, input):
        '''This input will be a tensor of frames extracted from a video'''
        x = self.Conv3dNetwork(input)
        x = self.norm(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.resnet(x)
        return x

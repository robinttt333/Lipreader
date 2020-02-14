import torch.nn as nn
from .conv3dNet import Conv3dNetwork
import torch.nn.functional as F
from .resnet import Resnet
import config


class Seq2SeqEncoder(nn.Module):
    '''This is the encoder class'''

    def __init__(self):
        super(Seq2SeqEncoder, self).__init__()
        self.Conv3dNetwork = Conv3dNetwork()
        self.norm = nn.BatchNorm3d(config.encoder["3dCNN"]["outputChannels"])
        self.pool = nn.MaxPool3d(config.encoder["pool"]["kernel"], stride=config.encoder["pool"]["stride"],
                                 padding=config.encoder["pool"]["padding"])
        self.resnet = Resnet()

    def forward(self, input):
        '''This input will be a tensor of frames extracted from a video'''
        x = self.Conv3dNetwork(input)
        x = self.norm(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.resnet(x)
        return x

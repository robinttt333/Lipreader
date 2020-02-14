import torch.nn as nn
import torch.nn.functional as F
import config


class Conv3dNetwork(nn.Module):
    '''This class implements the 3d convolution Network'''

    def __init__(self):
        super(Conv3dNetwork, self).__init__()
        self.conv3d = nn.Conv3d(
            config.image["channels"], config.frontend["3dCNN"]["outputChannels"], config.frontend["3dCNN"]["kernel"],
            config.frontend["3dCNN"]["stride"], config.frontend["3dCNN"]["padding"])

    def forward(self, input):
        input = self.conv3d(input)
        return input

import torch.nn as nn
import torch.nn.functional as F

class Conv3dNetwork(nn.Module):
    '''This class implements the 3d convolution Network'''
    def __init__(self,inputChannels,outputChannels,kernel,stride,padding):
        super(Conv3dNetwork,self).__init__()
        self.conv3d = nn.Conv3d(inputChannels,outputChannels,kernel,stride,padding)
        self.pool = nn.MaxPool3d((1,3,3),stride=(1,2,2),padding=(0,1,1))

    def forward(self,input):
        input = self.conv3d(input)
        input = F.relu(input)
        input = self.pool(input)
        return input
import torch.nn as nn
class Conv3dNetwork(nn.Module):
    def __init__(self,inputChannels,outputChannels,kernel,stride,padding):
        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding 
        self.conv3d = nn.Conv3d(inputChannels,outputChannels,kernel,stride,padding)

    def forward(self,x):
        return self.conv3d(x)
import torch.nn as nn
class Conv3dNetwork(nn.Module):
    '''This class implements the 3d convolution Network'''
    def __init__(self,inputChannels,outputChannels,kernel,stride,padding):
        super(Conv3dNetwork,self).__init__()
        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding 
        self.conv3d = nn.Conv3d(inputChannels,outputChannels,kernel,stride,padding)

    def forward(self,x):
        return self.conv3d(x)
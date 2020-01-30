import torch.nn as nn
from .conv3dNet import Conv3dNetwork
import torch.nn.functional as F
class Seq2SeqEncoder(nn.Module):
    '''This is the encoder class'''
    def __init__(self,paramsEncoder):
        super(Seq2SeqEncoder,self).__init__()
        self.Conv3dNetwork = Conv3dNetwork(paramsEncoder["inputChannels"],paramsEncoder["outputChannels"],
        paramsEncoder["kernel"],paramsEncoder["stride"],paramsEncoder["padding"])
        self.norm = nn.BatchNorm3d(paramsEncoder["outputChannels"])
        self.pool = nn.MaxPool3d(paramsEncoder["poolKernel"],stride=paramsEncoder["poolStride"],
        padding=paramsEncoder["poolPadding"])

    def forward(self,input):
        '''This input will be a tensor of frames extracted from a video'''
        x = self.Conv3dNetwork(input)
        print(x.shape)
        x = self.norm(x)
        x = F.relu(x)
        x = self.pool(x)
        return x
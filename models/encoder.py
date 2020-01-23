import torch.nn as nn
from .conv3dNet import Conv3dNetwork

class Seq2SeqEncoder(nn.Module):
    '''This is the encoder class'''
    def __init__(self,paramsEncoder):
        super(Seq2SeqEncoder,self).__init__()
        self.Conv3dNetwork = Conv3dNetwork(paramsEncoder["inputChannels"],paramsEncoder["outputChannels"],
        paramsEncoder["kernel"],paramsEncoder["stride"],paramsEncoder["padding"])

    def forward(self,input):
        '''This input will be a tensor of frames extracted from a video'''
        return self.fc2(self.fc1(input))

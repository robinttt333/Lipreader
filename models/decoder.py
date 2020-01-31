import torch.nn as nn
from .lstm import BidirectionalLSTM
class Seq2SeqDecoder(nn.Module):
    '''This is the decoder class'''
    def __init__(self,paramsDecoder):
        super(Seq2SeqDecoder,self).__init__()
        self.bidirectionalLSTM = BidirectionalLSTM(paramsDecoder)

    '''This takes in the output of the encoder ie a vector with fixed dimensions'''
    def forward(self,input):
        return self.bidirectionalLSTM(input)


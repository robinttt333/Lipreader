import torch.nn as nn
class Seq2SeqDecoder(nn.Module):
    '''This is the decoder class'''
    def __init__(self,paramsDecoder):
        super(Seq2SeqDecoder,self).__init__()
        pass
    '''This takes in the output of the encoder ie a vector with fixed dimensions'''
    def forward(self,input):
        return input


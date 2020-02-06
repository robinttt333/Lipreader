import torch.nn as nn
from .lstm import BidirectionalLSTM
from globalVariables import BACKEND_TYPE
from .temporalCNN import TemporalCNN

class Seq2SeqDecoder(nn.Module):
    '''This is the decoder class'''
    def __init__(self,paramsDecoder):
        super(Seq2SeqDecoder,self).__init__()
        if BACKEND_TYPE == "temporal CNN":
            self.model = nn.Sequential(
                TemporalCNN(paramsDecoder),
                nn.Linear(paramsDecoder["hiddenDimensions"],paramsDecoder["classes"]),
                nn.LogSoftmax(dim=1)
            )
        else :
            self.model = nn.Sequential(
                BidirectionalLSTM(paramsDecoder),
                nn.Linear(paramsDecoder["hiddenDimensions"]*2,paramsDecoder["classes"]),
                nn.LogSoftmax(dim=2)
            )
        
    '''This takes in the output of the encoder ie a vector with fixed dimensions'''
    def forward(self,input):
        return self.model(input)

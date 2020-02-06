import torch.nn as nn
from .lstm import BidirectionalLSTM
from .temporalCNN import TemporalCNN
"""Note that for lstm we use Nll over sequence and for temporal CNN we use CrossEntopy.
Since CrossEntropy adds softmax on its own and NLL does not we manually add a softmax layer in case
of lstm.
"""
class Seq2SeqDecoder(nn.Module):
    '''This is the decoder class'''
    def __init__(self,paramsDecoder):
        super(Seq2SeqDecoder,self).__init__()
        if paramsDecoder["backend_type"] == "temporal CNN":
            self.model = nn.Sequential(
                TemporalCNN(paramsDecoder),
                nn.Linear(paramsDecoder["hiddenDimensions"],paramsDecoder["classes"]),
            )
        elif paramsDecoder["backend_type"] == "lstm":
            self.model = nn.Sequential(
                BidirectionalLSTM(paramsDecoder),
                nn.Linear(paramsDecoder["hiddenDimensions"]*2,paramsDecoder["classes"]),
                nn.LogSoftmax(dim=2)
            )
        
    '''This takes in the output of the encoder ie a vector with fixed dimensions'''
    def forward(self,input):
        return self.model(input)

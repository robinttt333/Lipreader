import torch.nn as nn
from .Encoder import Seq2SeqEncoder
from .Decoder import Seq2SeqDecoder

class initLipArchitecture(nn.Module):
    '''This is the main class for the model.The model is based on a seq2seq
    ie it takes in an input sequence of video frames and finally converts them
    into a vecotor of fixed dimensions and this is then fed into an encoder'''

    def __init__(self,paramsEncoder,paramsDecoder):
        super(initLipArchitecture,self).__init__()
        self.Seq2SeqEncoder = Seq2SeqEncoder(paramsEncoder)
        self.Seq2SeqDecoder = Seq2SeqDecoder(paramsDecoder) 
    
    def forward(self,input):
        return self.Seq2SeqDecoder(self.Seq2SeqEncoder(input)) 

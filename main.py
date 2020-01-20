import torch
import torch.nn as nn

class Seq2SeqEncoder(nn.Module):
    '''This is the encoder class'''
    def __init__(self,paramsEncoder):
        self.fc1 = nn.Linear(0,0,bias=True)
        self.fc2 = nn.Linear(0,0,bias=True)

    def forward(self,input):
        '''This input will be a tensor of frames extracted from a video'''
        return self.fc2(self.fc1(input))

class Seq2SeqDecoder(nn.Module):
    '''This is the decoder class'''
    def __init__(self,paramsDecoder):
        self.fc1 = nn.Linear(0,0,bias=True)
        self.fc2 = nn.Linear(0,0,bias=True)
    '''This takes in the output of the encoder ie a vector with fixed dimensions'''
    def forward(self,input):
        return self.fc2(self.fc1(input))



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

if __name__ == "__main__":
    initLipArchitecture(0,0)    
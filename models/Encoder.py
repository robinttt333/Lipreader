import torch.nn as nn
class Seq2SeqEncoder(nn.Module):
    '''This is the encoder class'''
    def __init__(self,paramsEncoder):
        super(Seq2SeqEncoder,self).__init__()
        self.fc1 = nn.Linear(120,120,bias=True)
        self.fc2 = nn.Linear(120,120,bias=True)

    def forward(self,input):
        '''This input will be a tensor of frames extracted from a video'''
        return self.fc2(self.fc1(input))

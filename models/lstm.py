import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    def __init__(self,paramsDecoder):
        super(BidirectionalLSTM,self).__init__()
        self.biLSTM = nn.LSTM(paramsDecoder["inputFeatures"],paramsDecoder["hiddenDimensions"],
                    num_layers=paramsDecoder["lstmLayers"],batch_first=True,bidirectional=True) 

    def forward(self,input):
        lstmOutput,(hidden,cell) = self.biLSTM(input)
        return lstmOutput
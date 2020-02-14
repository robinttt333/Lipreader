import torch.nn as nn
import config


class BidirectionalLSTM(nn.Module):
    def __init__(self):
        super(BidirectionalLSTM, self).__init__()
        self.biLSTM = nn.LSTM(config.frontend["resnet"]["size"], config.backend["hiddenSize"],
                              num_layers=config.backend["lstm"]["layers"],
                              batch_first=config.backend["lstm"]["batchFirst"],
                              bidirectional=config.backend["lstm"]["bidirectional"])

    def forward(self, input):
        lstmOutput, (hidden, cell) = self.biLSTM(input)
        return lstmOutput

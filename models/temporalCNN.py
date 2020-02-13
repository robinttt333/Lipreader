import torch.nn as nn
import torch.nn.functional as F
import torch


class TemporalCNN(nn.Module):
    def __init__(self, paramsDecoder):
        super(TemporalCNN, self).__init__()
        """Wehave 2 conv layers here followed by a linear fully connected layer.The idea is
        that each conv layer doubles the channels and halves the number of frames and the pooling layer
        only halves the number of fraems.So if I/P is of the form [1 , 256 , 29] 
        each layer transforms as follows :
            [1 , 256 , 29] ---> [1 , 512 , 14] ---> [1 , 512 , 7] ---> [1, 1024, 3]
        """
        self.conv1 = nn.Conv1d(paramsDecoder["bn_size"], 2 * paramsDecoder["bn_size"],
                               paramsDecoder["conv1_kernel"], paramsDecoder["conv1_stride"])
        self.norm1 = nn.BatchNorm1d(paramsDecoder["bn_size"] * 2)

        self.pool1 = nn.MaxPool1d(
            paramsDecoder["max_pool1_kernel"], paramsDecoder["max_pool1_stride"])

        self.conv2 = nn.Conv1d(2 * paramsDecoder["bn_size"], 4 * paramsDecoder["bn_size"],
                               paramsDecoder["conv2_kernel"], paramsDecoder["conv2_stride"])
        self.norm2 = nn.BatchNorm1d(paramsDecoder["bn_size"] * 4)

        self.linear = nn.Linear(
            4*paramsDecoder["bn_size"], paramsDecoder["bn_size"])
        self.norm3 = nn.BatchNorm1d(paramsDecoder["bn_size"])

    def forward(self, input):
        transposed = input.transpose(1, 2).contiguous()
        output = self.conv1(transposed)
        output = self.norm1(output)
        output = F.relu(output)

        output = self.pool1(output)

        output = self.conv2(output)
        output = self.norm2(output)
        output = F.relu(output)

        output = output.mean(2)
        output = self.linear(output)
        """BatchNorm1d requires more than 1 as 0th element but we are passing
            [1,256] which gives error.So we comment it out for now. Can be rectified by using
            drop_last=True in dataloader"""
        # output = self.norm3(output)
        output = F.relu(output)
        return output

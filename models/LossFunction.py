import torch.nn as nn
from globalVariables import FRAME_COUNT
class NLLSequenceLoss(nn.Module):

    def __init__(self):
        super(NLLSequenceLoss, self).__init__()
        self.criterion = nn.NLLLoss()
    
    def forward(self, input, target):
        loss = 0.0
        transposed = input.transpose(0,1).contiguous()
        for i in range(0, FRAME_COUNT):
            loss += self.criterion(transposed[i], target)
        
        return loss
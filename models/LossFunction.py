import torch.nn as nn

class NLLSequenceLoss(nn.Module):

    def __init__(self):
        super(NLLSequenceLoss, self).__init__()
        self.criterion = nn.NLLLoss
    def forward(self, input, target):
        loss = 0.0
        transposed = input.transpose(0,1).contigous()

        for i in range(0, 29):
            loss += self.criterion(transposed[i], target)
        
        return loss
import torch.nn as nn
from .lstm import BidirectionalLSTM
from .temporalCNN import TemporalCNN
import config
"""Note that for lstm we use Nll over sequence and for temporal CNN we use CrossEntopy.
Since CrossEntropy adds softmax on its own and NLL does not we manually add a softmax layer in case
of lstm.
"""


class Backend(nn.Module):
    '''This is the backend class'''

    def __init__(self, stage):
        super(Backend, self).__init__()
        if stage == 1:
            self.model = nn.Sequential(
                TemporalCNN(),
                nn.Linear(config.backend["hiddenSize"],
                          config.backend["classes"]),
            )
        else:
            self.model = nn.Sequential(
                BidirectionalLSTM(),
                nn.Linear(config.backend["hiddenSize"]
                          * 2, config.backend["classes"]),
                nn.LogSoftmax(dim=2)
            )

    '''This takes in the output of the frontend ie a vector with fixed dimensions'''

    def forward(self, input):
        return self.model(input)

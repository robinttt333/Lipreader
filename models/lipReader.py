import torch.nn as nn
from .frontend import Frontend
from .backend import Backend
from .lossFunction import NLLSequenceLoss
import re
from .validatorFunctions import temporalCNNValidator, lstmValidator
import config
from torchsummary import summary


class Lipreader(nn.Module):
    '''This is the main class for the model.The model is based on a seq2seq
    ie it takes in an input sequence of video frames and finally converts them
    into a vecotor of fixed dimensions and this is then fed into an frontend'''

    def __init__(self, stage=1):
        super(Lipreader, self).__init__()
        self.Frontend = Frontend()
        self.Backend = Backend()
        self.stage = "Stage " + str(stage)

        if self.stage == "Stage 1":
            self.Loss = nn.CrossEntropyLoss()
            self.Validate = temporalCNNValidator
        else:
            self.Loss = NLLSequenceLoss()
            self.validate = lstmValidator

        def weights_init(m):
            classname = m.__class__.__name__
            # Here re fails for Conv3dNetwork
            if classname in ["Conv1d", "Conv2d", "Conv3d"]:
                m.weight.data.normal_(0.0, 0.02)
            elif re.search("BatchNorm[123]d", classname):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0)
            elif re.search("Linear", classname):
                m.bias.data.fill_(0)

        self.apply(weights_init)

    def forward(self, input):
        return self.Backend(self.Frontend(input))

    def getModelSummary(self):
        summary(self, input_size=(
            config.image["channels"], config.image["frames"], config.image["height"], config.image["width"],))

    def loss(self, input, target):
        return self.Loss(input, target)

    def validate(self, input, target):
        return self.Validate(input, target)

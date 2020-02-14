import torch.nn as nn
from .encoder import Seq2SeqEncoder
from .decoder import Seq2SeqDecoder
from .LossFunction import NLLSequenceLoss
import re
from .validatorFunctions import temporalCNNValidator, lstmValidator
import config


class Lipreader(nn.Module):
    '''This is the main class for the model.The model is based on a seq2seq
    ie it takes in an input sequence of video frames and finally converts them
    into a vecotor of fixed dimensions and this is then fed into an encoder'''

    def __init__(self):
        super(Lipreader, self).__init__()
        self.Seq2SeqEncoder = Seq2SeqEncoder()
        self.Seq2SeqDecoder = Seq2SeqDecoder()

        if config.backend["type"] == "temporal CNN":
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
        return self.Seq2SeqDecoder(self.Seq2SeqEncoder(input))

    def getModelDetails(self):
        for param in self.state_dict():
            print(param, self.state_dict()[param].shape)

    def loss(self, input, target):
        return self.Loss(input, target)

    def validate(self, input, target):
        return self.Validate(input, target)

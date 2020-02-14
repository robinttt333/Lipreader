from torchvision.models import resnet
import torch.nn as nn
import config


class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.model = self.initModel(
            config.frontend["resnet"]["model"], config.frontend["resnet"]["preTrain"])
        """By default the pytorch resnet models take in 3 channels but we have 64 so we need to modify 
        the first layer input to 64 instead of 3"""
        self.model.conv1 = nn.Conv2d(config.frontend["3dCNN"]["outputChannels"], 64, kernel_size=7, stride=2, padding=3,
                                     bias=False)

    def initModel(self, model, preTrain):
        if model == "18":
            return resnet.resnet18(preTrain, num_classes=config.frontend["resnet"]["size"])
        elif model == "34":
            return resnet.resnet34(preTrain, num_classes=config.frontend["resnet"]["size"])
        elif model == "50":
            return resnet.resnet50(preTrain, num_classes=config.frontend["resnet"]["size"])

    def forward(self, input):
        """ input shape is batch * channels * frames * height * width. We need to convert this
        5d tensor to a 4d tensor. So we first change its shape to batch * frames * channels * height * width. Then, 
        since the input needs to be a 4d tensor we concatenate the batch and frames dimensions into 1 dimension by doing
        transposed.reshape(-1,64,28,28).
        Finally we reshape the O/P back to batch * frames * 256. This 256 will be an encoded representation 
        corresponding to each frame of the video.
        """
        height, width = input.shape[3], input.shape[4]
        input = input.transpose(1, 2)
        vector4d = input.reshape(-1,
                                 config.frontend["3dCNN"]["outputChannels"], height, width)
        output = self.model(vector4d)
        output = output.view(-1,
                             config.image["frames"], config.frontend["resnet"]["size"])
        return output

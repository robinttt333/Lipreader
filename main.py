import torch
import torch.nn as nn
from models.lipReader import Lipreader
from data.dataset import LRWDataset
from globalVariables import IMAGE_CHANNELS,CONV3dOUTPUT_CHANNELS,CONV3D_PADDING,CONV3d_KERNEL,CONV3d_STRIDE

if __name__ == "__main__":
    '''The path variable stores the path to the data.
    Here we are only testing with a single file ie test.mp4 in this directory only and so we use "." 
    '''
    path = "."
    data = LRWDataset(path)
    paramsEncoder = {
        "inputChannels" : IMAGE_CHANNELS,
        "outputChannels" : CONV3dOUTPUT_CHANNELS,
        "stride" : CONV3d_STRIDE,
        "padding" : CONV3D_PADDING,
        "kernel" : CONV3d_KERNEL
    }
    paramsDecoder = ""
    lipreaderModel = Lipreader(paramsEncoder,paramsDecoder)
    lipreaderModel(data)
    print("Everything Working")
    
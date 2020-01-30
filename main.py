import torch
import torch.nn as nn
from models.lipReader import Lipreader
from data.dataset import LRWDataset
from globalVariables import (IMAGE_CHANNELS,CONV3dOUTPUT_CHANNELS,CONV3D_PADDING,
CONV3d_KERNEL,CONV3d_STRIDE,IMAGE_TRANSFORMS,BATCH_SIZE,SHUFFLE,FRONTEND_POOL_KERNEL,
FRONTEND_POOL_STRIDE,FRONTEND_POOL_PADDING)
from torch.utils.data import DataLoader

if __name__ == "__main__":
    '''The path variable stores the path to the data.
    Here we are only testing with a single file ie test.mp4 in this directory only and so we use "." 
    '''
    path = "."
    dataset = LRWDataset(path,IMAGE_TRANSFORMS)
    dataLoader =  DataLoader(dataset,batch_size = BATCH_SIZE,shuffle = SHUFFLE)
    paramsEncoder = {
        "inputChannels" : IMAGE_CHANNELS,
        "outputChannels" : CONV3dOUTPUT_CHANNELS,
        "stride" : CONV3d_STRIDE,
        "padding" : CONV3D_PADDING,
        "kernel" : CONV3d_KERNEL,
        "poolKernel" : FRONTEND_POOL_KERNEL,
        "poolStride" : FRONTEND_POOL_STRIDE,
        "poolPadding" : FRONTEND_POOL_PADDING
    }
    paramsDecoder = ""
    lipreaderModel = Lipreader(paramsEncoder,paramsDecoder)
    for i,batch in enumerate(dataLoader):
        lipreaderModel(batch[0])
    print("Everything Working")
    
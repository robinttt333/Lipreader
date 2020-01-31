import torch
import torch.nn as nn
from models.lipReader import Lipreader
from data.dataset import LRWDataset
from globalVariables import (IMAGE_CHANNELS,CONV3dOUTPUT_CHANNELS,CONV3D_PADDING,
CONV3d_KERNEL,CONV3d_STRIDE,IMAGE_TRANSFORMS,BATCH_SIZE,SHUFFLE,FRONTEND_POOL_KERNEL,
FRONTEND_POOL_STRIDE,FRONTEND_POOL_PADDING,RESNET_MODEL,PRE_TRAIN_RESNET,
ENCODER_REPRESENTATION_SIZE,LSTM_HIDDEN_SIZE,FRAME_COUNT,LSTM_LAYERS,NUM_CLASSES)
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
        "poolPadding" : FRONTEND_POOL_PADDING,
        "resnetModel" : RESNET_MODEL,
        "preTrain" : PRE_TRAIN_RESNET,
        "frames" : FRAME_COUNT
    }
    paramsDecoder = {
        "inputFeatures" : ENCODER_REPRESENTATION_SIZE,
        "hiddenDimensions" :  LSTM_HIDDEN_SIZE,
        "frames" : FRAME_COUNT,
        "lstmLayers" : LSTM_LAYERS,
        "classes" : NUM_CLASSES

    }
    lipreaderModel = Lipreader(paramsEncoder,paramsDecoder)
    for i,batch in enumerate(dataLoader):
        print(lipreaderModel(batch[0]).shape)
    print("Everything Working")
    
import torch
import torch.nn as nn
from models.lipReader import Lipreader
from data.dataset import LRWDataset
from globalVariables import (IMAGE_CHANNELS,CONV3dOUTPUT_CHANNELS,CONV3D_PADDING,
CONV3d_KERNEL,CONV3d_STRIDE,IMAGE_TRANSFORMS,BATCH_SIZE,SHUFFLE,FRONTEND_POOL_KERNEL,
FRONTEND_POOL_STRIDE,FRONTEND_POOL_PADDING,RESNET_MODEL,PRE_TRAIN_RESNET,
ENCODER_REPRESENTATION_SIZE,LSTM_HIDDEN_SIZE,FRAME_COUNT,LSTM_LAYERS,NUM_CLASSES,BACKEND_TYPE,
BN_SIZE,CONV1_KERNEL,CONV1_STRIDE,CONV2_KERNEL,CONV2_STRIDE,MAX_POOL1_KERNEL,MAX_POOL1_STRIDE
)
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
        "frames" : FRAME_COUNT,
        "backend_type" : BACKEND_TYPE

    }
    if BACKEND_TYPE == "temporal CNN":
        paramsDecoder = {
            "conv1_kernel" : CONV1_KERNEL,
            "conv1_stride" : CONV1_STRIDE,
            "conv2_kernel" : CONV2_KERNEL,
            "conv2_stride" : CONV2_STRIDE,
            "max_pool1_kernel" : MAX_POOL1_KERNEL,
            "max_pool1_stride" : MAX_POOL1_STRIDE,
            "bn_size" : BN_SIZE,
            "inputFeatures" : ENCODER_REPRESENTATION_SIZE,
            "hiddenDimensions" :  LSTM_HIDDEN_SIZE,
            "frames" : FRAME_COUNT,
            "lstmLayers" : LSTM_LAYERS,
            "classes" : NUM_CLASSES,
            "backend_type" : BACKEND_TYPE

            
    }
    else :
        paramsDecoder = {

        "inputFeatures" : ENCODER_REPRESENTATION_SIZE,
        "hiddenDimensions" :  LSTM_HIDDEN_SIZE,
        "frames" : FRAME_COUNT,
        "lstmLayers" : LSTM_LAYERS,
        "classes" : NUM_CLASSES,
        "backend_type" : BACKEND_TYPE

    }
    
    lipreader = Lipreader(paramsEncoder,paramsDecoder)
    for i,batch in enumerate(dataLoader):
        op = lipreader(batch[0])
    loss = lipreader.loss(op,torch.LongTensor(1).random_(0, 500))
    print("Current loss: ",loss.item())
    print("Everything Working")
    
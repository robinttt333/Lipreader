import cv2
import torch
from globalVariables import IMAGE_CHANNELS,IMAGE_HEIGHT,IMAGE_WIDTH,FRAME_COUNT

def extractFramesFromSingleVideo(video):
    frames = []
    video = cv2.VideoCapture(video)
    success,image = video.read()
    frames.append(torch.FloatTensor(image))
    '''The above step of specifying dtype is important because the weights are initialized by default 
    as float32 dtype and data is int type.Thus due to incompatibility between the 2 types we will get an
    error.We will apply a proper fix for this later.
    '''
    i = 1 
    while success and i<FRAME_COUNT: #We extract 29 frames in total from a single video 
        i+=1
        success,image = video.read()
        imageTensor = torch.FloatTensor(image)
        frames.append(imageTensor)
    return frames #list of n   256 * 256 * 3


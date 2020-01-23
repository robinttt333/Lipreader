import cv2
import torch
from globalVariables import IMAGE_CHANNELS,IMAGE_HEIGHT,IMAGE_WIDTH,FRAME_COUNT

def extractFramesFromSingleVideo(video):
    video = cv2.VideoCapture(video)
    success,image = video.read()
    '''We create a tensor of frames. Each frame is 256 * 256 * 3 
    ie h * w * c where h = height,w = width,c = channels in size. We stack them one top of another
    so that the final dimesions of frames is c * n * h * w where n=number of images.
    '''
    frames = torch.tensor(image,dtype=torch.float32).unsqueeze(dim=0)
    '''The above step of specifying dtype is important because the weights are initialized by default 
    as float32 dtype and data is int type.Thus due to incompatibility between the 2 types we will get an
    error.We will apply a proper fix for this later.
    '''
    frames = frames.reshape(IMAGE_CHANNELS,1,IMAGE_HEIGHT,IMAGE_WIDTH)
    i = 1 
    while success and i<FRAME_COUNT: #We extract 29 frames in total from a single video 
        i+=1
        success,image = video.read()
        imageTensor = torch.tensor(image,dtype=torch.float32).unsqueeze(0) #Tensor of size 1 * 256 * 256 * 3
        imageTensor = imageTensor.reshape(IMAGE_CHANNELS,1,IMAGE_HEIGHT,IMAGE_WIDTH)
        frames = torch.cat((frames,imageTensor),dim=1)
    print(frames.shape)
    print("Total %d frames extracted" % frames.shape[1])
    return frames #Tensor of n * 3 * 256 * 256


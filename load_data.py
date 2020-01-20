import glob
import cv2
import torch
def extractFramesFromSingleVideo(video):
    video = cv2.VideoCapture(video)
    success,image = video.read()
    '''We create a tensor of frames. Each frame is 256 * 256 * 3 
    ie h * w * c where h = height,w = width,c = channels in size. We stack them one top of another
    so that the final dimesions of frames is n * h * w * c where n=number of images.
    '''
    frames = torch.tensor(image).unsqueeze(dim=0)
    i = 1 
    while success and i<29: #We extract 29 frames in total from a single video 
        i+=1
        success,image = video.read()
        imageTensor = torch.tensor(image).unsqueeze(0) #Tensor of size 1 * 256 * 256 * 3
        frames = torch.cat((frames,imageTensor),dim=0)
        
    print("Total %d frames extracted" % frames.shape[0])
    return frames #Tensor of n * 256 * 256 * 3

def LoadData(path="."):
    videos = glob.glob(path + '/*.mp4') #Extarct all mp4 files in current dir 
    videoFrames = []
    for video in videos:
        videoFrames.append(extractFramesFromSingleVideo(video))
    return videoFrames
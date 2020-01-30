from torch.utils.data import Dataset
from load_data import extractFramesFromSingleVideo
import glob
import os
from globalVariables import FRAME_COUNT,IMAGE_HEIGHT,IMAGE_WIDTH
import torchvision.transforms as transforms
import torch

class LRWDataset(Dataset):
    def __init__(self,path,transforms=None):
        self.path = path
        self.transforms = transforms if transforms!=None else []
        self.videos = []
        self.labels = []
        self.initVideos(path)
        self.initLabels(path)
    
    def __getitem__(self,index):
        video = self.videos[index]
        path = os.path.join(self.path,video + '.mp4')
        processed = None
        frames = extractFramesFromSingleVideo(path)
        for i,frame in enumerate(frames):
            #print(frame.shape) # 256 * 256 * 3
            image = transforms.Compose(self.transforms)(frame)
            #print(image.shape) # 1 * 112 * 112
            if i==0:
                processed = image.unsqueeze(0)
            else :
                processed = torch.cat((processed,image.unsqueeze(0)),dim=1)
        return processed,self.labels[index]

    def __len__(self):
        return len(self.videos)

    def initLabels(self,path):
        self.labels = [0]
    
    def initVideos(self,path):
        videos = glob.glob(os.path.join(path,'*.mp4')) #Extarct all mp4 files in current dir 
        for video in videos:
            video = os.path.basename(video) #turns './test.mp4' to 'test.mp4' 
            name,extension = os.path.splitext(video) #returns test and .mp4
            self.videos.append(name)
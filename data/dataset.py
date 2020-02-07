from torch.utils.data import Dataset
from load_data import extractFramesFromSingleVideo
import glob
import os
from globalVariables import FRAME_COUNT,IMAGE_HEIGHT,IMAGE_WIDTH
import torchvision.transforms as transforms
import torch
import re
class LRWDataset(Dataset):
    def __init__(self,path,transforms=None):
        self.path = path
        self.transforms = transforms if transforms!=None else []
        self.samples = []
        self.init(path)
    
    def __getitem__(self,index):
        label,path = self.samples[index]
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
        return processed,label

    def __len__(self):
        return len(self.samples)

    def init(self,path):
        """Our videos are stored as follows.Inside the main folder there are 500 folders,
        1 for each label inside which are the files."""
        path = os.path.join(path,"data/videos/")
        labels = os.listdir(path)
        for i,label in enumerate(labels):
            dir = os.path.join(path,label)
            videos = os.listdir(dir)
            for video in videos:
                if video.find(".mp4") :
                    self.samples.append((i,os.path.join(dir,video)))

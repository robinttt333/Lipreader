from torch.utils.data import Dataset
from load_data import extractFramesFromSingleVideo
import glob
class LRWDataset(Dataset):
    def __init__(self,path):
        self.path = path
        self.videos = []
        self.labels = []
        self.initVideos(path)
        self.initLabels(path)
    
    def __getitem__(self,index):
        video = self.videos[index]
        return video
    
    def __len__(self):
        return len(self.videos)

    def initLabels(self,path):
        pass
    
    def initVideos(self,path):
        videos = glob.glob(path + '/*.mp4') #Extarct all mp4 files in current dir 
        videoFrames = []
        for video in videos:
            videoFrames.append(extractFramesFromSingleVideo(video))
        self.videos = videoFrames 
        
        

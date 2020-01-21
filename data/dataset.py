from torch.utils.data import Dataset

class LRWDataset(Dataset):
    def __init__(self,path):
        self.path = path
        self.videos = []
        self.labels = []
        self.initVideos(path)
        self.initLabels(path)
    
    def __getitem__(self,index):
        pass
    
    def __len__(self):
        return len(self.videos)

    def initLabels(self,path):
        pass
    
    def initVideos(self,path):
        pass
    
    

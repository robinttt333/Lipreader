from torch.utils.data import Dataset

class LRWDataset(Dataset):
    def __init__(self,path):
        self.path = path
        self.videos = []
        self.labels = []
        self.initVideos()
        self.initLabels()
    
    def __getitem__(self,index):
        pass
    
    def __len__(self):
        return len(self.videos)

    def initLabels(path):
        pass
    
    def initVideos(path):
        pass
    
    

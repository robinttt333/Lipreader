from data.dataset import LRWDataset
from torch.utils.data import DataLoader
from models.lipReader import Lipreader
class Trainer():
    def __init__(self,paramsEncoder,paramsDecoder,hyperParams,dataParams):
        self.lipreader = Lipreader(paramsEncoder,paramsDecoder)
        self.dataset = LRWDataset(dataParams["path"],dataParams["mode"],dataParams["transforms"])
        self.dataLoader =  DataLoader(self.dataset,batch_size = dataParams["batch_size"],
        shuffle = dataParams["shuffle"])
    
    def train(self):
        for _,batch in enumerate(self.dataLoader):    
            input,target = batch
            output = self.lipreader(input)
            loss = self.lipreader.loss(output,target)
            
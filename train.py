from data.dataset import LRWDataset
from torch.utils.data import DataLoader
from models.lipReader import Lipreader
from torch import optim
class Trainer():
    def __init__(self,paramsEncoder,paramsDecoder,hyperParams,dataParams):
        self.lipreader = Lipreader(paramsEncoder,paramsDecoder)
        self.dataset = LRWDataset(dataParams["path"],dataParams["mode"],dataParams["transforms"])
        self.dataLoader =  DataLoader(self.dataset,batch_size = dataParams["batch_size"],
        shuffle = dataParams["shuffle"])
        self.learningRate = hyperParams["learningRate"]
        self.momentum = hyperParams["momentum"]

    def train(self):
        optimizer = optim.SGD(self.lipreader.parameters(),lr = self.learningRate,
                    momentum = self.momentum)

        for _,batch in enumerate(self.dataLoader): 
            optimizer.zero_grad()   
            input,target = batch
            output = self.lipreader(input)
            loss = self.lipreader.loss(output,target)
            loss.backward()
            optimizer.step()
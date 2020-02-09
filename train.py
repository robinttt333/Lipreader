from data.dataset import LRWDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.lipReader import Lipreader
from torch import optim


class Trainer():
    def __init__(self, paramsEncoder, paramsDecoder, hyperParams, dataParams):
        # Check if gpu is available
        self.device = "gpu:0" if torch.cuda.is_available() else "cpu"
        self.lipreader = Lipreader(
            paramsEncoder, paramsDecoder)

        if torch.cuda.device_count() > 1:
            self.lipreader = nn.DataParallel(self.lipreader)
        self.lipreader = self.lipreader.to(self.device)
        print(self.lipreader)
        # print(next(self.lipreader.parameters()).is_cuda)
        self.dataset = LRWDataset(
            dataParams["path"], dataParams["mode"], dataParams["transforms"])
        self.dataLoader = DataLoader(self.dataset, batch_size=dataParams["batch_size"],
                                     shuffle=dataParams["shuffle"])
        self.learningRate = hyperParams["learningRate"]
        self.momentum = hyperParams["momentum"]

    def train(self):
        optimizer = optim.SGD(self.lipreader.parameters(), lr=self.learningRate,
                              momentum=self.momentum)

        for _, batch in enumerate(self.dataLoader):
            optimizer.zero_grad()
            input, target = batch
            input = input.to(self.device)
            target = target.to(self.device)
            output = self.lipreader(input)
            loss = self.lipreader.loss(output, target)
            loss.backward()
            optimizer.step()

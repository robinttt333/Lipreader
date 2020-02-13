from data.dataset import LRWDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from datetime import datetime, timedelta


class Trainer():
    def __init__(self, lipreader, hyperParams, dataParams):
        # Check if gpu is available
        self.device = "gpu:0" if torch.cuda.is_available() else "cpu"
        self.model = lipreader

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model).cuda()
        self.dataset = LRWDataset(
            dataParams["path"], "train", dataParams["transforms"])
        # set drop_last = True once the entire dataset is available
        self.dataLoader = DataLoader(self.dataset, batch_size=dataParams["batch_size"],
                                     shuffle=dataParams["shuffle"])
        self.learningRate = hyperParams["learningRate"]
        self.momentum = hyperParams["momentum"]

    def train(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.learningRate,
                              momentum=self.momentum)
        print("Started training at", datetime.now())
        for _, batch in enumerate(self.dataLoader):
            optimizer.zero_grad()
            input, target = batch
            input = input.to(self.device)
            target = target.to(self.device)
            output = self.model(input)

            print(self.model.validate(output, target))

            loss = self.model.loss(output, target)
            loss.backward()
            optimizer.step()

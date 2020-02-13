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
        self.batchSize = dataParams["batch_size"]

    def train(self, epoch):
        optimizer = optim.SGD(self.model.parameters(), lr=self.learningRate,
                              momentum=self.momentum)
        print("Started training at", datetime.now())

        correctOutputs = 0
        totalVideos = 0
        totalLoss = 0.0
        for _, batch in enumerate(self.dataLoader):
            optimizer.zero_grad()
            input, target = batch
            input = input.to(self.device)
            target = target.to(self.device)
            output = self.model(input)
            loss = self.model.loss(output, target)
            loss.backward()
            optimizer.step()

            correctOutputs += self.model.validate(output, target)
            totalVideos += len(batch)
            totalLoss += loss.data * len(batch)

        print(f"Epoch number {epoch+1} completed")
        print(f"Total samples: {totalVideos}")
        print(f"Correct samples: {correctOutputs}")
        print(f"Avg loss {totalLoss/totalVideos}")
        print(f"Avg acc {totalLoss/totalVideos}")

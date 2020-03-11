from data.dataset import LRWDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import config
from utils import countCorrectOutputs, saveStatsToCSV


class Trainer():
    def __init__(self, lipreader):
        # Check if gpu is available
        # self.device = "gpu:0" if torch.cuda.is_available() else "cpu"
        self.model = lipreader
        self.dataset = LRWDataset("train")
        # set drop_last = True once the entire dataset is available
        self.dataLoader = DataLoader(self.dataset, batch_size=config.data["batchSize"],
                                     shuffle=config.data["shuffle"])
        self.learningRate = config.training[self.model.stage]["learningRate"]
        self.momentum = config.hyperParams["momentum"]
        self.batchSize = config.data["batchSize"]

    def train(self, epoch):
        self.model = self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learningRate,
                              momentum=self.momentum)
        trainingStats = []
        for _, batch in enumerate(self.dataLoader):
            optimizer.zero_grad()
            input, target = batch
            input = input.cuda()
            target = target.cuda()
            output = self.model(input)
            correct = countCorrectOutputs(self.model.stage,
                                          target, output).item()
            loss = self.model.loss(output, target)
            trainingStat = {
                "Stage": self.model.stage,
                "Epoch": epoch+1,
                "Batch": _+1,
                "TrainingVideos":  input.shape[0],
                "CorrectTrainingOutputs": correct,
                "Loss": loss.item()
            }
            trainingStats.append(trainingStat)
            loss.backward()
            optimizer.step()
        saveStatsToCSV(
            trainingStats, epoch+1, "training", self.model.stage)

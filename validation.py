from data.dataset import LRWDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import config
from utils import saveStatsToCSV


class Validation():
    def __init__(self, lipreader):
        self.device = "gpu:0" if torch.cuda.is_available() else "cpu"
        self.model = lipreader

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model).cuda()
        self.validationDataset = LRWDataset("val")
        self.valiadtionDataLoader = DataLoader(self.validationDataset, batch_size=config.data["batchSize"],
                                               shuffle=config.data["shuffle"])

    def validate(self, epoch):
        self.model = self.model.eval()
        validationStats = []
        for _, batch in enumerate(self.valiadtionDataLoader):
            input, target = batch
            input = input.to(self.device)
            label = target.to(self.device)
            output = self.model(input)
            correct = self.model.validate(output, label)
            validationStat = {
                "Stage": self.model.stage,
                "Epoch": epoch+1,
                "Batch": _+1,
                "validationVideos": input.shape[0],
                "correctValidationOutputs": correct,
            }
            validationStats.append(validationStat)
        saveStatsToCSV(
            validationStats, epoch+1, "validation", self.model.stage)

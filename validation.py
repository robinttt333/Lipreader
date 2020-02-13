from data.dataset import LRWDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Validation():
    def __init__(self, lipreader, hyperParams, dataParams):
        self.device = "gpu:0" if torch.cuda.is_available() else "cpu"
        self.model = lipreader

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model).cuda()
        self.validationDataset = LRWDataset(
            dataParams["path"], "val", dataParams["transforms"])
        self.valiadtionDataLoader = DataLoader(self.validationDataset, batch_size=dataParams["batch_size"],
                                               shuffle=dataParams["shuffle"])

    def validate(self):
        correct = 0
        for _, batch in enumerate(self.valiadtionDataLoader):
            input, target = batch
            input = input.to(self.device)
            label = target.to(self.device)
            output = self.model(input)

            correct += self.model.validate(output, label)

        print("Number of correct outputs ", correct)
        print("Percentage of correct outputs",
              (correct/len(self.validationDataset))*100)

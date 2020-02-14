import config
from train import Trainer
from validation import Validation
from models.lipReader import Lipreader
import os
if __name__ == "__main__":
    lipreader = Lipreader()
    trainer = Trainer(lipreader)
    validator = Validation(lipreader)

    for epoch in range(config.training["completedEpochs"], config.training["epochs"]):
        trainer.train(epoch)
        validator.validate()

import config
from train import Trainer
from validation import Validation
from models.lipReader import Lipreader
from tqdm import tqdm
from datetime import datetime, timedelta
from savingAndLoading import saveModel, loadModel
import argparse
import os
if __name__ == "__main__":
    """https://docs.python.org/3.3/library/argparse.html"""
    """We store the last epoch in the trained model.So we can use that to set
    the starting point for begining the training.
    """
    parser = argparse.ArgumentParser(description='Running Lipnet')
    parser.add_argument('--load', type=str,
                        help='Name of the file containing the model')

    args = parser.parse_args()

    modelPath = args.load
    startEpoch = 1

    lipreader = Lipreader()
    if modelPath is not None:
        lipreader, lastEpoch = loadModel(lipreader, modelPath)
        startEpoch = lastEpoch + 1

    trainer = Trainer(lipreader)
    validator = Validation(lipreader)

    print("Started training at", datetime.now())
    with tqdm(total=config.training["epochs"]-startEpoch+1, desc="Epochs", position=0) as t:
        for epoch in range(startEpoch-1, config.training["epochs"]):
            trainer.train(epoch)
            validator.validate()
            t.update()
            saveModel(lipreader, epoch+1)

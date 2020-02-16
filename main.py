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
    parser = argparse.ArgumentParser(description='Running Lipnet')
    """For now we are adding only 2 command line args namely the start Epoch and the 
    path to the model.Clearly if startEpoch is more than 1 we must have a pre-trained
    model.But if start Epoch is 1 then we don't take a model path. 
    """
    parser.add_argument('startFromEpoch', type=int,
                        help='The last epoch that completed successfully')
    parser.add_argument('--load', type=str,
                        help='Path for the model to be loaded')

    args = parser.parse_args()

    startEpoch = args.startFromEpoch
    modelPath = args.load

    if startEpoch is not 1:
        if modelPath is None:
            raise Exception(
                "You need to provide model path if start Epoch is not 1")
        elif not os.path.exists(modelPath):
            raise Exception(f'No such file "{modelPath}" exists')
    elif startEpoch is 1 and modelPath is not None:
        raise Exception("Can't have a trained model on first epoch")

    if modelPath is not None:
        lipreader = loadModel(startEpoch-1)
    else:
        lipreader = Lipreader()
    trainer = Trainer(lipreader)
    validator = Validation(lipreader)

    print("Started training at", datetime.now())
    with tqdm(total=config.training["epochs"], desc="Epochs", position=0) as t:
        for epoch in range(startEpoch-1, config.training["epochs"]):
            trainer.train(epoch)
            validator.validate()
            t.update()
            saveModel(lipreader, epoch+1)

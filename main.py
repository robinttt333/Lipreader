import config
from train import Trainer
from validation import Validation
from models.lipReader import Lipreader
from tqdm import tqdm
from datetime import datetime, timedelta
from savingAndLoading import saveModel, loadModel
import argparse
import os
from utils import getStageFromFileName, getLastEpochFromFileName, checkIfFileExists, stageChangeRequired
from evaluate import evaluateVideo

if __name__ == "__main__":
    """https://docs.python.org/3.3/library/argparse.html"""
    """We store the last epoch in the trained model.So we can use that to set
    the starting point for begining the training.
    """
    parser = argparse.ArgumentParser(description='Running Lipnet')
    parser.add_argument('--load', type=str,
                        help='Name of the file containing the model')

    parser.add_argument('--evaluate', type=str,
                        help='Name of the file containing the model')

    args = parser.parse_args()

    fileName = args.load
    evaluate = args.evaluate
    if evaluate:
        if fileName:
            evaluateVideo(fileName, evaluate)
            exit(0)
        else:
            raise Exception("Please provide a fileName")
    startEpoch = 1
    stage = 1
    changeStage = False
    if fileName is not None:
        if not checkIfFileExists(fileName):
            raise ValueError(
                "No such file exists in the specified path...Please see the 'dir' option under savingAndLoading in the config and ensure that your file is present there")
        lastEpoch = getLastEpochFromFileName(fileName)
        startEpoch = lastEpoch + 1
        stage = getStageFromFileName(fileName)
        if stageChangeRequired(stage, lastEpoch):
            changeStage = True
            startEpoch = 1
            stage += 1
        lipreader = Lipreader(stage)
        lipreader = loadModel(lipreader, fileName, changeStage)
    else:
        lipreader = Lipreader()
    trainer = Trainer(lipreader)
    validator = Validation(lipreader)
    totalEpochs = config.training["Stage "+str(stage)]["epochs"]
    print("Started training at", datetime.now())
    with tqdm(total=totalEpochs-startEpoch+1, desc="Epochs", position=0) as t:
        for epoch in range(startEpoch-1, totalEpochs):
            trainer.train(epoch)
            validator.validate(epoch)
            t.update()
            saveModel(lipreader, epoch+1)
        print(f"Successfully completed training of Stage {stage}")

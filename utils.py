
import config
import os
import csv


def getStageFromFileName(fileName):
    """Filename is of the form Epoch1_1.pt where the 1 after _ represents the stage"""
    return int(fileName.split("_")[1].split(".")[0])


def getLastEpochFromFileName(fileName):
    """Filename is of the form Epoch1_1.pt where the 1 before _ represents the stage"""
    return int(fileName.split("_")[0][5:])


def checkIfFileExists(fileName):
    dir = config.savingAndLoading["dir"]
    file = os.path.join(os.path.curdir, dir, fileName)
    if not os.path.exists(file):
        raise ValueError(
            "No such file exists in the specified path...Please see the 'dir' option under savingAndLoading in the config and ensure that your file is present there")


def stageChangeRequired(stage, epoch):
    stage1Epochs = config.training['Stage 1']["epochs"]
    stage2Epochs = config.training['Stage 2']["epochs"]
    stage3Epochs = config.training['Stage 3']["epochs"]
    return True if (stage == 1 and epoch == stage1Epochs) or (stage == 2 and epoch == stage2Epochs) or (stage == 3 and epoch == stage3Epochs) else False


def countCorrectOutputs(stage, target, output):
    """In stage 1 we have O/P as n*500 and in the other 2 stages as
        n * 29 * 500.So we convert the O/P of second stage to first by adding
        data over first dimension to get a n*500 vector and then take the index
        of max value out of the 500 labels to get a 1d vector of size n.
        Then we just need to compare the matching vals b/w the outputs and labels
        and sum the true(1) values as false values are set to 0 by default.
    """
    if stage != 'Stage 1':
        output = output.sum(1)
    return (target == output.argmax(1)).sum()


def saveStatsToCSV(data, epoch, mode, stage):
    fileName = os.path.join(getPath(epoch, stage),
                            getUniqueName(epoch, stage, mode, "csv"))
    with open(fileName, 'w') as csvfile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def getUniqueName(epoch, stage, mode="", type="pt"):
    filename = f"Epoch{epoch}_{stage[-1]}{mode}.{type}"
    return filename


def getPath(epoch, stage):
    dir = config.savingAndLoading["dir"]
    if not os.path.exists(dir):
        os.mkdir(dir)
    path = os.path.join(os.path.curdir, dir)
    dirName = f"Epoch{epoch}_{stage[-1]}"
    if not os.path.exists(os.path.join(path, dirName)):
        os.mkdir(os.path.join(path, dirName))
    return os.path.join(path, dirName)

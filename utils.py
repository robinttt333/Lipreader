
import config
import os


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
    return True if (stage == 1 and epoch == 1) or (stage == 2 and epoch == 5) and (stage == 3 and epoch == 30) else False

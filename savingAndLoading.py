import config
import os
import torch
from models.lipReader import Lipreader
from utils import getLastEpochFromFileName, stageChangeRequired, getStageFromFileName, changeStage


def getUniqueName(epoch, stage):
    filename = f"Epoch{epoch}_{stage}.pt"
    return filename


def saveModel(model, epoch):
    # print(next(iter(model.state_dict().items())))
    # print(next(iter(model.named_parameters())))
    """Above 2 statements show that state_dict does not contain grad status of the params but 
    names params does.So we save grad status separately.If we do model.load_state_dict all grads
    will be reset to True by default.This is not what we want as we wish to train backend 
    and frontend separately ie we wish to freeze some layers by making requires_grad = False.  
    """
    dir = config.savingAndLoading["dir"]
    if not os.path.isdir(dir):
        os.mkdir(dir)
    grad_states = {}
    for param, tensor in model.named_parameters():
        grad_states[param] = tensor.requires_grad

    state = {
        "state_dict": model.state_dict(),
        "grad_states": grad_states,
    }
    stage = int(model.stage.split()[1])
    file = os.path.join(os.path.curdir, dir, getUniqueName(epoch, stage))
    torch.save(state, file)
    print(f"Model saved as Epoch{epoch}_{stage}.pt")


def updateGradStatus(model, state):
    for param, tensor in model.named_parameters():
        tensor.requires_grad_(state["grad_states"][param])
    return model


def loadModel(model, fileName):
    """file is the full path of file and fileName is only its name"""
    dir = config.savingAndLoading["dir"]
    file = os.path.join(os.path.curdir, dir, fileName)
    state = torch.load(file)
    epoch = getLastEpochFromFileName(fileName)
    stage = getStageFromFileName(fileName)
    """
        stage changes take place when :
        1) If model is in stage 1 and completes 30 epochs
        2) If model is in stage 2 and completes 5 epochs
        3) If model is  in stage 3 and completes 30 epochs

        This check is required so that the appropriate layers are frozen and unfrozen
    """
    if stageChangeRequired(stage, epoch):
        changeStage(model, stage)
    else:
        print(f"Loading model with last completed epoch as : {epoch}")
        model.load_state_dict(state["state_dict"])
        return updateGradStatus(model, state)

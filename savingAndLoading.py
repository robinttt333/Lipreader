import config
import os
import torch
from models.lipReader import Lipreader
from utils import getLastEpochFromFileName, stageChangeRequired, getStageFromFileName, getPath, getUniqueName


def saveModel(model, epoch):
    # print(next(iter(model.state_dict().items())))
    # print(next(iter(model.named_parameters())))
    """Above 2 statements show that state_dict does not contain grad status of the params but 
    names params does.So we save grad status separately.If we do model.load_state_dict all grads
    will be reset to True by default.This is not what we want as we wish to train backend 
    and frontend separately ie we wish to freeze some layers by making requires_grad = False.  
    """
    grad_states = {}
    for param, tensor in model.named_parameters():
        grad_states[param] = tensor.requires_grad

    state = {
        "state_dict": model.state_dict(),
        "grad_states": grad_states,
    }
    file = os.path.join(getPath(epoch, model.stage),
                        getUniqueName(epoch, model.stage))
    torch.save(state, file)
    print(f"Model saved as Epoch{epoch}_{model.stage[-1]}.pt")


def updateGradStatus(model, state):
    for param, tensor in model.named_parameters():
        tensor.requires_grad_(state["grad_states"][param])
    return model


def loadModel(model, fileName, change=False):
    """file is the full path of file and fileName is only its name"""
    dir = config.savingAndLoading["dir"]
    file = os.path.join(os.path.curdir, dir, fileName.split(".")[0], fileName)
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
    if change:
        print(f"Updating the stage from {stage} ---> {stage+1}")
        return changeStage(model, stage, state)
    print(f"Loading model with last completed epoch as : {epoch}")
    model.load_state_dict(state["state_dict"])
    return updateGradStatus(model, state)


def changeStage(model, stage, pretrainedDict):
    currentStateDict = model.state_dict()
    pretrainedDict = {k: v for k,
                      v in pretrainedDict.items() if k in currentStateDict}
    currentStateDict.update(pretrainedDict)
    model.load_state_dict(currentStateDict)

    if stage == 1:
        for param, tensor in model.Frontend.named_parameters():
            tensor.requires_grad_(False)
    elif stage == 2:
        for param, tensor in model.Frontend.named_parameters():
            tensor.requires_grad_(True)
    return model

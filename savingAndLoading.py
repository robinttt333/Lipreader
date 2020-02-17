import config
import os
import torch
from datetime import datetime
from models.lipReader import Lipreader


def getUniqueName(epoch):
    filename = f"Epoch{epoch}.pt"
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
        "lastEpoch": epoch
    }
    file = os.path.join(os.path.curdir, dir, getUniqueName(epoch))
    torch.save(state, file)
    print(f"Model saved as Epoch{epoch}.pt")


def updateGradStatus(model, state):
    for param, tensor in model.named_parameters():
        tensor.requires_grad_(state["grad_states"][param])
    return model


def loadModel(model, fileName):
    """First check if the file exists"""
    dir = config.savingAndLoading["dir"]
    file = os.path.join(os.path.curdir, dir, fileName)
    if not os.path.exists(file):
        raise ValueError(
            "No such file exists in the specified path...Please see the 'dir' option under savingAndLoading in the config")

    state = torch.load(file)
    epoch = state["lastEpoch"]
    print(f"Loading model with last completed epoch as : {epoch}")
    model.load_state_dict(state["state_dict"])
    return updateGradStatus(model, state), epoch

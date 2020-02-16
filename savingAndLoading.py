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
    print("Saving your model...")
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
    print(f"Model saved after epoch {epoch}")


def updateGradStatus(model, state):
    for param, tensor in model.named_parameters():
        tensor.requires_grad_(state["grad_states"][param])
    return model


def loadModel(epoch):
    print(f"Loading model after {epoch} epochs")
    dir = config.savingAndLoading["dir"]
    file = os.path.join(os.path.curdir, dir, getUniqueName(epoch))
    model = Lipreader()
    state = torch.load(file)
    model.load_state_dict(state["state_dict"])
    return updateGradStatus(model, state)

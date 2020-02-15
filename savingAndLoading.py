import config
import os
import torch
from datetime import datetime
from models.lipReader import Lipreader


def getUniqueName(epoch):
    filename = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename += f"-Epoch{epoch+1}.pt"
    return filename


def saveModel(model, epoch):
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
    print(f"Model saved after epoch {epoch+1}")


def updateGradStatus(model, state):
    for param, tensor in model.named_parameters():
        tensor.requires_grad_(state["grad_states"][param])


def loadModel(epoch):
    print(f"Loading model after {epoch} epochs")
    dir = config.savingAndLoading["dir"]
    file = os.path.join(os.path.curdir, dir, getUniqueName(epoch))
    model = Lipreader()
    state = torch.load(file)
    model.load_state_dict(state["state_dict"])
    return updateGradStatus(model, state)

from models.lipReader import Lipreader
from utils import checkIfFileExists
import torch
import config
import os
from load_data import extractFramesFromSingleVideo
import torchvision.transforms as Transforms


def evaluateVideo(modelFile, video):

    video = os.path.join(os.path.curdir, video)
    if checkIfFileExists(modelFile):
        if os.path.exists(video):
            dir = config.savingAndLoading["dir"]
            file = os.path.join(os.path.curdir, dir,
                                modelFile.split(".")[0], modelFile)
            state = torch.load(file)
            model = Lipreader()
            model.load_state_dict(state["state_dict"])
            model.eval()
            transforms = config.data["transforms"] if config.data["transforms"] != None else [
            ]
            frames = extractFramesFromSingleVideo(video)
            for i, frame in enumerate(frames):
                # print(frame.shape) # 256 * 256 * 3
                image = Transforms.Compose(transforms)(frame)
                # print(image.shape) # 1 * 112 * 112
                if i == 0:
                    processed = image.unsqueeze(0)
                else:
                    processed = torch.cat(
                        (processed, image.unsqueeze(0)), dim=1)

            output = model(processed.unsqueeze(0)).data.max(dim=1)
            probability, label = round(
                output.values.item(), 3), output.indices.item()
            print(
                f"The predicted word is afternoon with probability {probability}")
        else:
            raise Exception("No such video file exists")
    else:
        raise Exception("No such model file exists")

import torch
import torch.nn as nn

from models import initLipArchitecture

if __name__ == "__main__":
    paramsEncoder = ""
    paramsDecoder = ""
    initLipArchitecture(paramsEncoder,paramsDecoder)
    print("Everything Working")
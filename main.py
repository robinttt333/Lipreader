import torch
import torch.nn as nn

from models.lipReader import Lipreader

if __name__ == "__main__":
    paramsEncoder = ""
    paramsDecoder = ""
    Lipreader(paramsEncoder,paramsDecoder)
    print("Everything Working")
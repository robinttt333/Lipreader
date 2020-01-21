import torch
import torch.nn as nn
from models.lipReader import Lipreader
from data.dataset import LRWDataset
if __name__ == "__main__":
    '''The path variable stores the path to the data.
    Here we are only testing with a single file ie test.mp4 in this directory only and so we use "." 
    '''
    path = "."
    data = LRWDataset(path)
    paramsEncoder = ""
    paramsDecoder = ""
    Lipreader(paramsEncoder,paramsDecoder)
    print("Everything Working")
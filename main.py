import torch
import torch.nn as nn
from load_data import LoadData
from models.lipReader import Lipreader

if __name__ == "__main__":
    '''The path variable stores the path to the data.
    Here we are only testing with a single file ie test.mp4 in this directory only and so we use "." 
    '''
    path = "."
    data = LoadData(path)
    paramsEncoder = ""
    paramsDecoder = ""
    Lipreader(paramsEncoder,paramsDecoder)
    print("Everything Working")
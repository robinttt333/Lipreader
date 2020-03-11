'''This is a file created to hold our global variables'''
import torchvision.transforms as transforms
import torch

#GPU

gpuAvailable = torch.cuda.is_available()
deviceCount = torch.cuda.device_count()

# Image features
image = {
    "height": 256,
    "width": 256,
    "channels": 1,
    "frames": 29,
}

# frontend settings
frontend = {
    "3dCNN": {
        "outputChannels": 64,
        "kernel": (5, 7, 7),
        "stride": (1, 2, 2),
        "padding": (2, 3, 3),
    },
    "pool": {
        "kernel": (1, 3, 3),
        "stride": (1, 2, 2),
        "padding": (0, 1, 1)
    },
    "resnet": {
        "model": "34",
        "preTrain": False,
        "shuffle": True,
        "size": 256
    }

}

# Dataloader and dataset settings
data = {
    "batchSize": 10,
    "shuffle": True,
    "path": ".",
    "transforms": [
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        # Cropping to be done to 112 * 112 based on the research paper
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize([0.4161, ], [0.1688, ]),
    ]
}
# Backend Settings
backend = {
    "lstm": {
        "layers": 2,
        "bidirectional": True,
        "batchFirst": True
    },
    "temporal CNN": {
        "conv1Kernel": 2,
        "conv1Stride": 2,
        "maxPool1Kernel": 2,
        "maxPool1Stride": 2,
        "conv2Kernel": 2,
        "conv2Stride": 2,
    },
    "hiddenSize": 256,
    "classes": 500
}
# Hyperparams
hyperParams = {
    "momentum": .9
}

# Training
"""We have 3 stages of training as mentioned in the paper and 
https://github.com/tstafylakis/Lipreading-ResNet which are as follows:
"""
training = {
    "Stage 1":
    {
        "Backend": "temporal CNN",
        "learningRate": .003,
        "epochs": 30,
    },
    "Stage 2":
    {
        "Backend": "lstm",
        "learningRate": .003,
        "epochs": 5,
    },
    "Stage 3": {
        "Backend": "lstm",
        "learningRate": .0005,
        "epochs": 1,
    }
}
# Saving and Loading the model
savingAndLoading = {
    "dir": "savedStates",

}

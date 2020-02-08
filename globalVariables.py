'''This is a file created to hold our global variables'''
import torchvision.transforms as transforms
#Image features
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_CHANNELS = 1
FRAME_COUNT = 29 
IMAGE_TRANSFORMS = [
    transforms.ToPILImage(),
    transforms.CenterCrop((112,112)),  #Cropping to be done to 112 * 112 based on the research paper
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.4161,],[0.1688,]),
    ]

#3d CNN settings
CONV3dOUTPUT_CHANNELS = 64
CONV3d_KERNEL = (5,7,7)
CONV3d_STRIDE = (1,2,2) 
CONV3D_PADDING = (2,3,3)
FRONTEND_POOL_KERNEL = (1,3,3)
FRONTEND_POOL_STRIDE = (1,2,2)
FRONTEND_POOL_PADDING = (0,1,1)

#Resnet settings
RESNET_MODEL = "34"
PRE_TRAIN_RESNET = False
SHUFFLE = True
ENCODER_REPRESENTATION_SIZE = 256

#LSTM settings
LSTM_HIDDEN_SIZE = 256
LSTM_LAYERS = 2

#Dataloader settings
BATCH_SIZE = 10

#Backend Settings
BACKEND_TYPE = "temporal CNN"
BN_SIZE = 256
CONV1_KERNEL = 2
CONV1_STRIDE = 2
MAX_POOL1_KERNEL = 2
MAX_POOL1_STRIDE = 2
CONV2_KERNEL = 2
CONV2_STRIDE = 2
NUM_CLASSES = 500

#Training Hyperparams
EPOCHS = 1
COMPLETED_EPOCHS = 0
LEARNING_RATE = .003
MOMENTUM = .9
'''This is a file created to hold our global variables'''
import torchvision.transforms as transforms

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_CHANNELS = 1
FRAME_COUNT = 29 
CONV3dOUTPUT_CHANNELS = 64
CONV3d_KERNEL = (5,7,7)
CONV3d_STRIDE = (1,2,2) 
CONV3D_PADDING = (2,3,3)
IMAGE_TRANSFORMS = [
    transforms.ToPILImage(),
    transforms.CenterCrop((112,112)),  #Cropping to be done to 112 * 112 based on the paper
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.4161,],[0.1688,]),
    ]
FRONTEND_POOL_KERNEL = (1,3,3)
FRONTEND_POOL_STRIDE = (1,2,2)
FRONTEND_POOL_PADDING = (0,1,1)
BATCH_SIZE = 10
SHUFFLE = True
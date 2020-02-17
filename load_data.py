import cv2
import torch
import config


def extractFramesFromSingleVideo(video):
    frames = []
    video = cv2.VideoCapture(video)
    success, image = video.read()
    frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    """PIL uses RGB format but cv2 uses BGR format for 3 channel images and since in our
    transforms we are converting each frame into PIL image before doing anything on it 
    this step becomes becessary ie using cv2.COLOR_BGR2RGB.
    """
    i = 1
    # We extract 29 frames in total from a single video
    while success and i < config.image["frames"]:
        i += 1
        success, image = video.read()
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return frames  # list of n   256 * 256 * 3

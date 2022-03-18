# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""

import glob

import os

import time

from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
# import torch
# import torch.nn.functional as F
from PIL import ExifTags

# from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

from utils.augmentations import  letterbox
from utils.general import (LOGGER, check_requirements, clean_str)
# from utils.torch_utils import torch_distributed_zero_first

# Parameters
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))  # DPP
NUM_THREADS = min(8, os.cpu_count())  # number of multiprocessing threads

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

def LoadImages(path, img_size=640, stride=32, auto=True):
    #  image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`

    # assert key.split('.')[-1].lower() in IMG_FORMATS, 'Image format not supported'

    img0 = path
    assert img0 is not None, f'Image Not Found {path}'

    # Padded resize
    img = letterbox(img0, img_size, stride=stride, auto=auto)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    s = f'image 1/1 food: '

    return img, img0, s

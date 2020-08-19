import torch
from project_moduls.Unet.Models import ResNetUNet_RoadSeg
from project_moduls.Unet.Dataset import TargetImages
from Utils import SegmentImage , GetParkingSpotCoords ,Classify,GetVacantParking
import project_moduls.geo.geo as geo
import numpy as np
import cv2
import PIL
import matplotlib.pyplot as plt
from transform import *


if __name__ == '__main__':
    # define the unet model 
    img_name = '2.png'
    write_str = 'hh.png'
    path = 'project_moduls\\Unet\\res_unet_IOU_MSE_454.pth'
    dataset_dir = 'test_images'
    index = 3
    to_print = True

    with torch.no_grad():
        GetVacantParking({},{})


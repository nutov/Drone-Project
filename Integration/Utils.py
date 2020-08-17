import torch
from project_moduls.Unet.Models import ResNetUNet_RoadSeg
from project_moduls.Unet.Dataset import TargetImages,GetTransform_data
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from transform import *
import project_moduls.geo.geo as geo
from project_moduls.Classifier.Model import Classifier
import PIL
def SegmentImage(path = 'project_moduls\\Unet\\res_unet_IOU_MSE_454.pth',dataset_dir = 'test_images',
                write_str = 'hh.png'):
    PATH = path
    segmentor = ResNetUNet_RoadSeg()
    segmentor.load_state_dict(torch.load(PATH))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    segmentor = segmentor.to(device).eval()
    
    # define target images datasets
    dataset = TargetImages(dataset_dir)
    target_dataloader = torch.utils.data.DataLoader(dataset)
    
    # for count,im in enumerate(target_dataloader):
    im = next(iter(target_dataloader))
    im  = im.to(device).detach()
    seg = segmentor.predict_real_image(im)
    # TODO: improve 
    im_np = seg.detach().cpu().squeeze(dim=0).squeeze(dim=0).numpy()
    
    mask = np.array([im_np,im_np,im_np])
    mask = mask.transpose(1,2,0)
    mask[mask >= 0.4] = 255
    mask[mask < 0.4] = 0
    #mask = 255*mask
    mask = mask.astype(np.uint8)
    
    #plt.imshow(mask)
    #plt.show()
    #print(f'{mask.dtype}')
    #print(f'{type(mask)}')
    #print(f'{mask.shape}')
    #print(mask.shape)
    cv2.imwrite(write_str, mask)

def GetParkingSpotCoords(index = 3,dataset_dir = 'test_images',img_name = '2.png',to_print = True,write_str= 'hh.png'):

    mask = cv2.imread(write_str)

    coord_list = geo.GetCoords(mask)
    pts = np.array(eval(str(coord_list[index])),dtype = "float32")
    img = cv2.imread(os.path.join(dataset_dir,img_name))
    wraped = four_point_transform(img,pts )
    if to_print:
        print(type(mask))
        print(mask.dtype)
        print(coord_list[index])
        geo.show_one(mask,'org')
        geo.show_one(wraped,'cropped')
    return wraped


def Classify(img:torch.tensor,model_path = 'project_moduls\\Classifier\\classifier.pth'):
    if not isinstance(img,np.ndarray) or not isinstance(img,PIL.Image) or not isinstance(img,torch.tensor):
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Classifier().load_state_dict(torch.load(model_path)).eval().to(device)
    if isinstance(img,np.ndarray):
        img = torch.from_numpy(img)
        img = img.unsqueeze(dim=0)
        if len(img.shape) < 4:
            img = torch.nn.functional.interpolate(img,size = (224,224))

    elif isinstance(img,PIL.Image):
        trans = GetTransform_data()
        img = trans(img)
    
    
    
    img = img.to(device)
    output = model(img)
    return output
        


    



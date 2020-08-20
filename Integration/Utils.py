import torch
import torch.nn as nn
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
from torchvision import transforms

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

def GetParkingSpotCoords(mask:np.ndarray,img,to_print = False):
    
    #mask = cv2.imread(write_str)

    coord_list = geo.GetCoords(mask)
    for index,pts in enumerate(coord_list):
        #img = cv2.imread(os.path.join(dataset_dir,img_name))
        pts = np.array(eval(str(pts)),dtype = "float32")
        wraped = four_point_transform(img,pts )
        if to_print:
            print(type(mask))
            print(mask.dtype)
            print(coord_list[index])
            geo.show_one(mask,'org')
            geo.show_one(wraped,'cropped')
        yield wraped ,pts


def Classify(img,model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #model = Classifier()
    #model.load_state_dict(torch.load(model_path))
    #model = model.to(device).eval()


    
    img = torch.tensor(img,dtype = torch.float32).to(device)
    img /= 255.0
    img[0,:] -= torch.tensor([0.485],dtype = torch.float32).to(device)
    img[1,:] -= torch.tensor([0.456],dtype = torch.float32).to(device)
    img[2,:] -= torch.tensor([0.406],dtype = torch.float32).to(device)

    img[0,:] /= torch.tensor([0.229],dtype = torch.float32).to(device)
    img[1,:] /= torch.tensor([0.224],dtype = torch.float32).to(device)
    img[2,:] /= torch.tensor([0.225],dtype = torch.float32).to(device)
     
    img = img.unsqueeze(dim=0)

    img = torch.nn.functional.interpolate(img,size = (224,224),align_corners = True,mode = 'bilinear')

    #print('img shape ',img.shape)  
    output = model(img)
    output =  torch.sigmoid(output)
    return output
        


    
def GetVacantParking(unet_cfg:dict,classifier_cfg:dict):

    PATH = unet_cfg.get('path','project_moduls\\Unet\\res_unet_IOU_MSE_454.pth')
    dataset_dir = unet_cfg.get('dataset_dir','test_images')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    to_print = True

    segmentor = ResNetUNet_RoadSeg()
    segmentor.load_state_dict(torch.load(PATH))
    segmentor = segmentor.to(device).eval()

    empty_c = (0,255,0)
    occupied_c = (0,0,255)
    
    model_path = classifier_cfg.get('model_path','project_moduls\\Classifier\\classifier.pth')
    model = Classifier()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device).eval()

    # define target images datasets
    dataset = TargetImages(dataset_dir)
    target_dataloader = torch.utils.data.DataLoader(dataset)
    for count,img in enumerate(target_dataloader):

        img  = img.to(device).detach()
        seg = segmentor.predict_real_image(img)
        im_np = seg.detach().cpu().squeeze(dim=0).squeeze(dim=0).numpy()
        mask = np.array([im_np,im_np,im_np])
        mask = mask.transpose(1,2,0)
        mask[mask >= 0.4] = 255
        mask[mask < 0.4] = 0
        mask = mask.astype(np.uint8)
        # TODO : work directly without reading and writing to a png file 
        #---------------------------------------------------------------
        cv2.imwrite(f'temp\\mask_{count}.png', mask)
        mask = cv2.imread(f'temp\\mask_{count}.png')
        img = cv2.imread('test_images\\2.png',cv2.COLOR_BGR2RGB)
        #---------------------------------------------------------------
        count_oc = 0
        for wraped ,pts in GetParkingSpotCoords(mask,img):
            #wraped = cv2.cvtColor(wraped, cv2.COLOR_BGR2RGB)
            wraped = wraped.transpose((2,0,1))
            #temp = wraped[0,:]
            #wraped[0,:] = wraped[2,:]
            #wraped[2,:] = temp
            res = Classify(wraped,model)
            res = res.detach().cpu().numpy()
            #img_ = img
            #img = img*255
            img = img.astype(np.uint8)
            
            if res[0] < 0.5:
                #print(f'empty \n {pts}')
                cv2.fillPoly(img,[pts.astype(int)],empty_c)
            else:
                cv2.fillPoly(img,[pts.astype(int)],occupied_c)
                count_oc = count_oc+1
                #print(f'occupied \n {pts}')
        print(count_oc)
        plt.imshow(img)
        plt.show()
        cv2.imwrite(f'results\\{count}.png',img)


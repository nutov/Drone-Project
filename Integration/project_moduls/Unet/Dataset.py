# -*- coding: utf-8 -*-
"""
Created on Sun May 10 22:54:49 2020

@author: notov
"""
import pandas as pd
import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
import cv2
import glob
import re
import numpy as np
import shutil

class SupervisedDataSet(torch.utils.data.Dataset):
    
    
    def __init__(self,
                 image_folder:str = 'patch_dataset/train/images',
                 label_folder:str= 'patch_dataset/train/labels'):

            
        self.image_folder = image_folder
        self.label_folder = label_folder
        
        self.images = os.listdir(self.image_folder)
        self.labels = os.listdir(self.label_folder)
        self.transform_data = GetTransform_data()
        self.transform_label = GetTransform_label()
        
    def __getitem__(self, index):
        x = Image.open(os.path.join(self.image_folder,self.images[index]))
        x = x.convert('RGB')
        
        y = Image.open(os.path.join(self.label_folder,self.labels[index]))
        y = y.convert('1')
        
        x = self.transform_data(x)
        y = self.transform_label(y)
        x.requires_grad = True
        
        y[y >= 0.5] = 1.0
        y[y < 0.5] = 0.0
        y.requires_grad = True
        return x,y
        
    def __len__(self):
        return len(self.images)
    
        



def GetTransform_data():
    custom_transforms = []
    

    custom_transforms.append(torchvision.transforms.Resize(size=(224,224)))
    custom_transforms.append(torchvision.transforms.ToTensor())
    custom_transforms.append(torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))


        
    return torchvision.transforms.Compose(custom_transforms)

def GetTransform_label():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.Resize(size=(224,224)))
    custom_transforms.append(torchvision.transforms.ToTensor())

        
        
    return torchvision.transforms.Compose(custom_transforms)


# this class is for loading the target images we wish to plot our model
# final results 
class TargetImages(torch.utils.data.Dataset):        
    def __init__(self,image_folder:str = 'target_images',size:tuple = (720,1280)):
        
        self.image_folder = image_folder
        self.images = os.listdir(self.image_folder)
        #(768,768)
        #(720,1280)
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size = size),
                                                         torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def __getitem__(self, index):
        x = Image.open(os.path.join(self.image_folder,self.images[index]))
        x = x.convert('RGB')
        return self.transform(x)        
    
    
    def __len__(self):
        return len(self.images)    




def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)




def create_dataloader_dict(image_folder_train:str,label_folder_train:str,
                            image_folder_val:str,label_folder_val:str,
                            batch_size=8,shuffle=True):
    '''
    creates a dataloader dictionary
    '''
    dataset_train = SupervisedDataSet(image_folder = image_folder_train,
                                label_folder = label_folder_train)

    dataset_val = SupervisedDataSet(image_folder = image_folder_val,
                                    label_folder = label_folder_val)  


    data_loader_train = torch.utils.data.DataLoader(dataset_train, 
                                                    batch_size=batch_size,
                                                    shuffle=shuffle)

    data_loader_val = torch.utils.data.DataLoader(dataset_val, 
                                                  batch_size=batch_size,
                                                  shuffle=shuffle)  

    data_loader_dict = {'train':data_loader_train,
                        'val': data_loader_val}  

    return data_loader_dict




def show_example_dataset():
    pass

def show_target_image():
    pass

def show_target_res():
    pass


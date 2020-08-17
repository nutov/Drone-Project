# -*- coding: utf-8 -*-
"""
Created on Sun May 10 23:43:44 2020

@author: notov
"""
from typing import NamedTuple, List, Iterator, Tuple, Union, Callable, Iterable
from PIL import Image,ImageDraw
import numpy as np


import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn 
#from torch_scatter import scatter_add
from Models import *
from Result_Analysis import *


def TrainUnet(dataloaders:dict,unet_model:nn.Module,loss_func:nn.Module,num_of_epochs:int = 25,loss_str = "MSE"):
    """
    This function trains unet from scratch and validate the models 
    trains by a specific loss function
    on a validation set
    also implemented early stop 

    Parameters
    ----------
    dataloaders : dict
        contains the train and validation dataloaders.
    unet_model : ResNetUNet
        unet architecture model.
    num_of_epochs : int, optional
        number of epochs to run. The default is 25.

    Returns
    -------
    metrics_train_dict,metrics_val_dict.
    """
    smooth = 0.01
    #loss_str = loss_func.__name__
    
    # final results on validation dataset 
    #'ACC':[],
    metrics_val_dict = {'IOU':[],
                         
                         'LOSS':[]}
    
    # final results on train dataset 
    #'ACC':[],
    metrics_train_dict = {'IOU':[],
                         
                         'LOSS':[]}

    cuda_flag = torch.cuda.is_available()
    best_iou = -np.Inf
    best_acc = -np.Inf
    
    if cuda_flag:
        unet_model = unet_model.cuda()
        
    optimizer = optim.SGD(unet_model.parameters(),lr = 1e-3,momentum=0.9)
    lr_sched = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    for epoch in range(num_of_epochs):
        
        # single epoch results on validation dataset 
        #'ACC':0.0
        metrics_epoch_val_dict = {'IOU':0.0}
        
        # single epoch results on train dataset 
        metrics_epoch_train_dict = {'IOU':0.0}
        
        train_loss = 0.0
        val_loss = 0.0
        unet_model = unet_model.train(True)
        
        #----------------------------------------------------------------------
        #---------------------------batch--------------------------------------
        #----------------------------------------------------------------------
        for x,y in dataloaders['train']:
            if cuda_flag:
                x = x.cuda()
                y = y.cuda()
            
               
            
            optimizer.zero_grad()         
            output = unet_model(x)            
            loss = loss_func(output,y)
            
            #pred_sig = unet_model.predict(x) 
            # saving loss and computing metrics
            train_loss += loss.item()
           
            metrics_epoch_train_dict['IOU'] += IOUMetric((output > 0.5).int(),y.int()).item()
            
            #metrics_epoch_train_dict['ACC'] += (float((pred_sig > 0.5) == y).sum())/(y.shape[2]*y.shape[3]*y.shape[0]*y.shape[1])

            loss.backward()
            optimizer.step()
               
        unet_model = unet_model.eval()
        
        for x,y in dataloaders['val']:
            with torch.no_grad():
                if cuda_flag:
                    x = x.cuda()
                    y = y.cuda()
    
                          
                output = unet_model(x) 
                 
                loss = loss_func(output,y)
                
                # saving loss and computing metrics
                val_loss += loss.item()
                metrics_epoch_val_dict['IOU'] += IOUMetric((output > 0.5).int(),y.int()).item()
                
                
        #----------------------------------------------------------------------        
        # save train and validation data        
        metrics_val_dict['IOU'].append(float(metrics_epoch_val_dict['IOU'])/float(len(dataloaders['val'])))
                
        metrics_train_dict['IOU'].append(float(metrics_epoch_train_dict['IOU'])/float(len(dataloaders['train'])))

        metrics_val_dict['LOSS'].append(float(val_loss)/float(len(dataloaders['val'])))
        
        metrics_train_dict['LOSS'].append(float(train_loss)/float(len(dataloaders['train'])))
        
        # save best model by each metric        
        if best_iou < metrics_val_dict['IOU'][-1]:
            best_iou = metrics_val_dict['IOU'][-1]
            # save model
            length = len(dataloaders['val'])
            f_name = f'res_unet_IOU_{loss_str}_{length}_{epoch}'
            SaveModel(f_name,unet_model)
            

           
        
        lr_sched.step()
        
    return metrics_train_dict, metrics_val_dict





#------------------------------------------------------------------------------
#-----------------------------Loss Functions-----------------------------------
#------------------------------------------------------------------------------
# we shoul swnd logits to the loss functions 
def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    pred = torch.sigmoid(pred)
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def BCE_Logits(pred, target):
    # TODO; maybe add weights
    criterion = torch.nn.BCEWithLogitsLoss()
    return criterion(pred, target)

def Combined_loss(pred, target):
    return 0.5*(dice_loss(pred, target) + BCE_Logits(pred, target))

def MSE(pred, target):
    criterion = torch.nn.MSELoss( reduction='mean')
    pred = pred.contiguous()
    pred = torch.sigmoid(pred)
    target = target.contiguous() 
    return criterion(pred, target)
    
#------------------------------------------------------------------------------
#-------------------------------  Metrics   ------------------------------------
#------------------------------------------------------------------------------

def IOUMetric(mask:torch.Tensor,g_t:torch.Tensor,smooth:float = 0.01):
    '''
    calculates the maen IOU metric over the batch     
    input should be in batch form ie: (b_d,1,H,W)
    Parameters
    ----------
    mask : torch.Tensor
        DESCRIPTION.
    g_t : torch.Tensor
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    mask_pred = mask.clone()
    mask_pred = (mask_pred > 0.5).int()
    inersection = (mask_pred & g_t.int()).sum(dim=(1,2,3)) + smooth
    union = (mask_pred | g_t.int()).sum(dim=(1,2,3)) + smooth
    res = inersection.float()/union.float()
    res = res.mean()
    
    return res 
    
    
def IOUMetricByClass(mask:torch.Tensor,g_t:torch.Tensor,smooth:float = 0.01):
    '''
    calculates the maen IOU metric over the batch     
    input should be in batch form ie: (b_d,1,H,W)
    Parameters
    ----------
    mask : torch.Tensor
        DESCRIPTION.
    g_t : torch.Tensor
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    mask_pred = mask.clone()
    mask_pred = (mask_pred > 0.5).int()
    inersection = (mask_pred & g_t.int()).sum(dim=(2,3))
    union = (mask_pred | g_t.int()).sum(dim=(2,3))
    res = inersection.float()/union.float()
    res = res.mean(dim=0)
    
    return res 
    
    






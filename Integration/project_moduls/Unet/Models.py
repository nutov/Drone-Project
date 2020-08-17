# -*- coding: utf-8 -*-
"""
Created on Sun May 10 23:00:58 2020

@author: notov
"""

import torch
from torch import nn
from torch.nn import functional as F

from torchvision import models
import torchvision
  
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# another optional version for unet 
class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(self, in_channels: int, channels: list, kernel_sizes: list,
                 batchnorm=True, dropout=0.,relu_flag = False,batch_norm_first = True):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
        convolution in the block. The length determines the number of
        convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
        be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
        convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
        Zero means don't apply dropout.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        self.main_path, self.shortcut_path = None, None
        self.relu_flag = relu_flag
        # Done: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  the main_path, which should contain the convolution, dropout,
        #  batchnorm, relu sequences, and the shortcut_path which should
        #  represent the skip-connection.
        #  Use convolutions which preserve the spatial extent of the input.
        #  For simplicity of implementation, we'll assume kernel sizes are odd.
        # ====== YOUR CODE: ======
        main_layers = []
        
        # constructing the input layer 
        # we assume kernel sizes are odd so to preserve spacial dimentions we 
        # padd with the kernel size divided by 2
        padding = kernel_sizes[0]//2
        main_layers.append(nn.Conv2d(in_channels,channels[0],kernel_sizes[0],padding = padding))
        if dropout > 0:
            main_layers.append(nn.Dropout2d(dropout))
        if batchnorm ==True:    
            main_layers.append(nn.BatchNorm2d(channels[0]))
        main_layers.append(nn.ReLU())
        
        for idx in range(len(channels)-1):
            padding = kernel_sizes[idx+1]//2
            main_layers.append(nn.Conv2d(channels[idx],channels[idx +1],kernel_sizes[idx+1],padding = padding))

            if idx < len(channels)-2:    
                main_layers.append(nn.ReLU())
                if dropout > 0:
                    main_layers.append(nn.Dropout2d(dropout))
                
                if batchnorm ==True:    
                    main_layers.append(nn.BatchNorm2d(channels[idx + 1]))
            
        if channels[-1] != in_channels:
            self.shortcut_path = nn.Sequential(nn.Conv2d(in_channels,channels[-1],kernel_size = 1,bias=False)
            )            
        else:    
            self.shortcut_path = nn.Sequential()
             
            
        self.main_path = nn.Sequential(*main_layers)
        # ========================

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        if self.relu_flag:
            out = torch.relu(out)
        
        return out

    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
class ResidualBlock_2(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(self, in_channels: int, channels: list, kernel_sizes: list,
                 batchnorm=True, dropout=0.,relu_flag = False,batch_norm_first = True):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
        convolution in the block. The length determines the number of
        convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
        be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
        convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
        Zero means don't apply dropout.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        self.main_path, self.shortcut_path = None, None
        self.relu_flag = relu_flag
        # Done: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  the main_path, which should contain the convolution, dropout,
        #  batchnorm, relu sequences, and the shortcut_path which should
        #  represent the skip-connection.
        #  Use convolutions which preserve the spatial extent of the input.
        #  For simplicity of implementation, we'll assume kernel sizes are odd.
        # ====== YOUR CODE: ======
        main_layers = []
        
        # constructing the input layer 
        # we assume kernel sizes are odd so to preserve spacial dimentions we 
        # padd with the kernel size divided by 2
        padding = kernel_sizes[0]//2
        
        if dropout > 0:
            main_layers.append(nn.Dropout2d(dropout))
        if batchnorm ==True:    
            main_layers.append(nn.BatchNorm2d(in_channels))
        main_layers.append(nn.ReLU())
        
        main_layers.append(nn.Conv2d(in_channels,channels[0],kernel_sizes[0],padding = padding))
        for idx in range(len(channels)-1):
            padding = kernel_sizes[idx+1]//2
            

            if idx < len(channels)-2:    
                
                if dropout > 0:
                    main_layers.append(nn.Dropout2d(dropout))
                
                if batchnorm ==True:    
                    main_layers.append(nn.BatchNorm2d(channels[idx + 1]))
                    
                main_layers.append(nn.ReLU())
                main_layers.append(nn.Conv2d(channels[idx],channels[idx +1],kernel_sizes[idx+1],padding = padding))
                
        if channels[-1] != in_channels:
            self.shortcut_path = nn.Sequential(nn.Conv2d(in_channels,channels[-1],kernel_size = 1,bias=False)
            )            
        else:    
            self.shortcut_path = nn.Sequential()
             
            
        self.main_path = nn.Sequential(*main_layers)
        # ========================

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        if self.relu_flag:
            out = torch.relu(out)
        
        return out
  
    
#------------------------------------------------------------------------------    

class UpPath(nn.Module):
    def __init__(self,in_dim,filter_num,is_deconve = False):
        super().__init__()
        if is_deconve:
            self.up = nn.ConvTranspose2d(in_dim,in_dim,kernel_size=4)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv = ResidualBlock_2(in_dim,[filter_num]*3,[3]*3)
    def forward(self,x,x_cat):
        x = self.up(x)
        x = torch.cat((x,x_cat),dim=1)
        x = self.conv(x) 
        return x
    
#------------------------------------------------------------------------------
# building encoder decoder models for the unet         
#------------------------------------------------------------------------------
# DownSample is with max pooling in paper is with stride
class Encoder(nn.Module):
    def __init__(self,in_dim,filter_num):
        super().__init__()
        self.in_dim = in_dim
        self.filter_num = filter_num
        
        self.e_layer_1 = ResidualBlock(in_dim,[filter_num]*2,[3]*2)
        self.max_pool_1 = nn.MaxPool2d(kernel_size = 2)
        self.e_layer_2 = ResidualBlock_2(filter_num,[filter_num*2]*3,[3]*3)
        self.max_pool_2 = nn.MaxPool2d(kernel_size = 2)
        self.e_layer_3 = ResidualBlock_2(filter_num*2,[filter_num*4]*3,[3]*3)
        self.max_pool_3 = nn.MaxPool2d(kernel_size = 2)
        
    def forward(self,x):
        x_1 = self.e_layer_1(x)
        x_2 = self.max_pool_2(self.e_layer_2(x_1))
        x_3 = self.max_pool_3(self.e_layer_3(x_2))
        return x_1,x_2,x_3
        
class Decoder(nn.Module):
    def __init__(self,up_dim,is_deconve = False):
        super().__init__()
        self.filter_num = 256
        self.d_layer_3 = UpPath(up_dim + self.filter_num, self.filter_num, is_deconve = is_deconve)
        self.d_layer_2 = UpPath(self.filter_num + self.filter_num//2 , self.filter_num//2 , is_deconve = is_deconve)
        self.d_layer_1 = UpPath(self.filter_num//2 + self.filter_num//4 , self.filter_num//4 , is_deconve = is_deconve)
        
        self.out_layer = nn.Conv2d(self.filter_num//4 , 1 , 1)
        self.sig = nn.Sigmoid()
        
    def forward(self,x_bridge,x_1,x_2,x_3):
        
        x = self.d_layer_3(x_bridge,x_3)
        x = self.d_layer_2(x,x_2)
        x = self.d_layer_1(x,x_1)
        
        x = self.sig(self.out_layer(x))
        return x
        
         
#------------------------------------------------------------------------------    
# ResUnet implemented by the paper Road Extraction by Deep Residual U-Net
#------------------------------------------------------------------------------
        
class ResNetUNet_RoadSeg(nn.Module):
    def __init__(self,in_dim = 3,filter_num = 64,up_dim = 512,is_deconve = False):
        super().__init__()
        #----------------------------------------------------------------------
        # encoder architecture
        #----------------------------------------------------------------------
        self.encoder = Encoder(in_dim,filter_num)
        #----------------------------------------------------------------------
        # bridge architecture
        #----------------------------------------------------------------------
        self.bridge = ResidualBlock_2(filter_num*4
                                      ,[up_dim,up_dim,up_dim],[3,3,3])
        self.maxpool = nn.MaxPool2d(kernel_size = 2)
        #----------------------------------------------------------------------
        # decoder architecture
        #----------------------------------------------------------------------       
        self.decoder = Decoder(up_dim)
        
    def forward(self,x):
        x_1,x_2,x_3 = self.encoder(x)
        #print(x_1.shape,x_2.shape,x_3.shape)
        
        x_bridge = self.bridge(x_3)
        x_bridge = self.maxpool(x_bridge)
        mask = self.decoder(x_bridge,x_1,x_2,x_3)
        return mask
        
    def predict_real_image(self,im_full_size):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        original_shape = im_full_size.shape
        # Deivide into patches
        col_num = 5
        row_num = 3
        x = torch.zeros((col_num*row_num,3,original_shape[2]//3 , original_shape[3]//5))
        dim_x,dim_y = original_shape[2]//3 , original_shape[3]//5
        # devide image into patches 
        for row in range(row_num):
            for col in range(col_num):
                x[row + col*row_num,:] = im_full_size[0,:,dim_x*row : dim_x*(row+1),dim_y*col : dim_y*(col+1)]
        
        if torch.cuda.is_available():
            x = x.cuda()
        x = F.interpolate(x,size = (224,224)
                          ,mode = 'bilinear'
                          ,align_corners = True)        

        mask = self.forward(x) 

        mask = F.interpolate(mask,size = (original_shape[2]//3 , original_shape[3]//5)
                             ,mode = 'bilinear'
                             ,align_corners = True)
        
        
        mask_restored = torch.zeros((1,1,dim_x*3,dim_y*5))
        
        for row in range(row_num):
            for col in range(col_num):
                mask_restored[0,:,dim_x*row : dim_x*(row+1),dim_y*col : dim_y*(col+1)]  =\
                mask[row + col*row_num,:]        

        return mask_restored   
        
def main():        
    x = torch.randn((1,3,720,1280),device = 'cuda')
    
    model = ResNetUNet_RoadSeg().cuda()
    
    with torch.no_grad():
        model = model.eval()
        z = model.predict_real_image(x)
    return z
        
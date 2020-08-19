import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.optim.optimizer import Optimizer
import torchvision.models as models
from torch.optim import lr_scheduler


# This module is a pre-trained classifier for the classification score 
class Classifier(nn.Module):
    """ 
    This module is a pre-trained classifier for the classification score.
      
    Attributes: 
        net (ResNet): A resnet pretrained architecture. 
        device (str): A string describing if a GPU is available.
    """ 
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(pretrained=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)
               
        self._FreezeModel()

        self.num_ftrs = self.net.fc.in_features  

        self.net.fc = nn.Linear(self.num_ftrs, 1)  
        print(self.net)
        self.to(self.device) 


    def forward(self,x):
        return self.net(x)
        
    def _FreezeModel(self):
        '''
        This function sets the requires_grad atrribute of the 
        mask rcnn model to false, hence it disables the backpropogation 
        ability for this model.

        Returns
        -------
        None.
        '''
        for p in self.net.parameters():
            p.requires_grad = False
    
    def SaveModel(self):
        pass

    def LoadModel(self):
        pass







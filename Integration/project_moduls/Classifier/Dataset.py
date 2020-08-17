from torchvision import datasets, models, transforms
import torch
import os

def GetDataTransformDict():
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    return data_transforms



def GetDataLoaderDict(data_dir:str = 'Parking',batch_size = 8):
    data_transforms = GetDataTransformDict()
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x]) for x in ['train', 'val']}
    #Create a dictionary that contians the data loader
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                batch_size=batch_size,
                                                shuffle=True) for x in ['train', 'val']}

    return dataloaders         

def GetTestDataLoader(data_dir:str = 'Parking',batch_size = 8):
    trans_ = transforms.Compose([transforms.Resize((224,224))
                                ,transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                                )
    image_datasets = datasets.ImageFolder(os.path.join(data_dir,'test'),trans_) 

    dataloaders =  torch.utils.data.DataLoader(image_datasets, 
                                                batch_size=batch_size,
                                                shuffle=True) 
    return dataloaders






import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim.optimizer import Optimizer
import torchvision.models as models
from torch.optim import lr_scheduler
import numpy as np
import torch.optim as optim
from Model import Classifier

def TrainClassifier(dataloaders:dict,model:Classifier
                    ,criterion:nn.Module,test_loader = None
                    ,num_of_epochs:int = 25):
    
    
    best_acc = -np.Inf
    metrics_val_dict = {'ACC':[],
                         
                         'LOSS':[]}

    metrics_train_dict = {'ACC':[],
                         
                         'LOSS':[]}   


    metrics_test_dict = {'ACC':[],
                         
                         'LOSS':[]}

    optimizer = optim.Adam(model.parameters(),lr = 1e-3)
    lr_sched = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.4)
    
    
    for _ in range (num_of_epochs):
        
        #Reset the correct to 0 after passing through all the dataset
        correct = 0
        total_train = 0
        total_trian_loss = 0.0
        total_val_loss = 0.0
        total_test_loss = 0.0
        total_test = 0
        model = model.train()
        for images,labels in dataloaders['train']:
            #ind = np.arange(images.shape[0])

            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
                
            optimizer.zero_grad()
            outputs = model(images).squeeze(dim=1)
            #ind_max = torch.argmax(outputs,dim=1)
            #logits = outputs[ind,ind_max]
            loss = criterion(outputs, labels.double())
            total_trian_loss += loss
            loss.backward()
            optimizer.step()  
            #_, predicted = torch.max(outputs, 1) 
            predicted = torch.sigmoid(outputs) > 0.5

            total_train += images.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = correct /float(total_train) 
        metrics_train_dict['ACC'].append(train_acc)
        metrics_train_dict['LOSS'].append(total_trian_loss.item()/len(dataloaders['train']))

        model = model.eval()  
        with torch.no_grad():
            correct_test = 0
            correct = 0
            total = 0
            for images, labels in dataloaders['val']:

                if torch.cuda.is_available():
                    
                    images = images.cuda()
                    labels = labels.cuda()

                #ind = np.arange(images.shape[0])
                outputs = model(images).squeeze(dim=1)
                
                #ind_max = torch.argmax(outputs,dim=1)
                
                #logits = outputs[ind,ind_max]
                
                loss = criterion(outputs, labels.double())

                total_val_loss += loss
                #_, predicted = torch.max(outputs.data, 1)
                predicted = torch.sigmoid(outputs) > 0.5
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            if test_loader is not None:
                for images, labels in test_loader:

                    if torch.cuda.is_available():
                        images = images.cuda()
                        labels = labels.cuda()

                    #ind = np.arange(images.shape[0])
                    outputs = model(images).squeeze(dim=1)
                    
                    #ind_max = torch.argmax(outputs,dim=1)

                    #logits = outputs[ind,ind_max]
                
                    loss = criterion(outputs, labels.double())
                    total_test_loss += loss
                    #_, predicted = torch.max(outputs.data, 1)
                    predicted = torch.sigmoid(outputs) > 0.5
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()


                
        metrics_val_dict['ACC'].append(correct/float(total))
        metrics_val_dict ['LOSS'].append(total_val_loss.item()/len(dataloaders['train']))

        metrics_test_dict['ACC'].append(correct_test/float(total_test))
        metrics_test_dict ['LOSS'].append(total_test_loss.item()/len(test_loader))


        # saves best model       
        if best_acc < metrics_val_dict['ACC'][-1]:
            best_acc = metrics_val_dict['ACC'][-1]
            # save model
            
            f_name = f'classifier'
            SaveModel(f_name,model)

        lr_sched.step()

    return metrics_train_dict , metrics_val_dict ,metrics_test_dict








def SaveModel(f_name:str,model):
    torch.save(model.state_dict(), f_name + '.pth')
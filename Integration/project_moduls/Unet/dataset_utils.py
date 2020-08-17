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

import matplotlib.pyplot as plt 


        
def DatasetPartition():

    image_test_folder = 'patch_dataset/test/images'
    label_test_folder= 'patch_dataset/test/labels'
    images_test = sorted_aphanumeric(os.listdir(image_test_folder))
    labels_test = sorted_aphanumeric(os.listdir(label_test_folder))   

    image_train_folder = 'patch_dataset/train/images'
    label_train_folder= 'patch_dataset/train/labels'
    images_train = sorted_aphanumeric(os.listdir(image_train_folder))
    labels_train = sorted_aphanumeric(os.listdir(label_train_folder))   

    image_validation_folder = 'patch_dataset/validation/images'
    label_validation_folder= 'patch_dataset/validation/labels'
    images_validation = sorted_aphanumeric(os.listdir(image_validation_folder))
    labels_validation = sorted_aphanumeric(os.listdir(label_validation_folder))   

    

    # Move a file from the directory d1 to d2
    #shutil.move('/Users/billy/d1/xfile.txt', '/Users/billy/d2/xfile.txt')
    test_len = len(images_test)
    p_1 = int(test_len*0.6)
    p_2 = int(test_len*0.7)
    test_ind_to_move = np.split(np.random.permutation(test_len),[p_1,p_2])

    train_len = len(images_train)
    p_1 = int(train_len*0.1)
    p_2 = int(train_len*0.8)
    train_ind_to_move = np.split(np.random.permutation(train_len),[p_1,p_2])


    val_len = len(images_validation)
    p_1 = int(val_len*0.3)
    p_2 = int(val_len*0.8)
    val_ind_to_move = np.split(np.random.permutation(val_len),[p_1,p_2])
    
    (count_train , count_val , count_remain) = MoveToDest(folder = '/test',image_folder=images_test,label_folder= labels_test
    ,ind_list = test_ind_to_move)
    (count_train , count_val , count_remain) = MoveToDest(folder = '/train',image_folder=images_train,label_folder= labels_train,
    ind_list = train_ind_to_move,count = (count_train , count_val , count_remain))
    
    (count_train , count_val , count_remain) = MoveToDest(folder = '/validation',image_folder=images_validation,label_folder= labels_validation,
    ind_list = val_ind_to_move,count = (count_train , count_val , count_remain))
    




def MoveToDest(folder:str,ind_list:list,image_folder:list,label_folder:list,count:tuple = (0,0,0)):
    # move datasets after partition from orig to dest
    orig = "E:/study/project_drone/Unet/patch_dataset"
    dest = "E:/study/project_drone/Unet/new_patch_dataset"

    count_train , count_val , count_remain = count
    
    

    for idx in ind_list[0]:
        shutil.move(orig + folder + '/images/'+image_folder[idx], dest+'/train/images/'+image_folder[idx])
        shutil.move(orig+ folder + '/labels/'+ label_folder[idx], dest+'/train/labels/'+label_folder[idx])
        if image_folder[idx] !=str(count_train)+'.png':
            os.rename(dest+ '/train/images/' +image_folder[idx] , dest+'/train/images/'+str(count_train)+'.png')
            os.rename(dest+'/train/labels/'+label_folder[idx],dest+'/train/labels/'+str(count_train)+'.png')
        count_train = count_train + 1
        
        
    for idx in ind_list[1]:
        shutil.move(orig+ folder + '/images/'+image_folder[idx], dest+'/validation/images/'+image_folder[idx])
        shutil.move(orig+ folder + '/labels/'+label_folder[idx], dest+'/validation/labels/'+label_folder[idx])
        if image_folder[idx] != str(count_val)+'.png':
            os.rename(dest+ '/validation/images/'+image_folder[idx],dest+'/validation/images/'+str(count_val)+'.png')
            os.rename(dest+ '/validation/labels/'+label_folder[idx],dest+'/validation/labels/'+str(count_val)+'.png')
        count_val = count_val + 1
    
    for idx in ind_list[2]:
        shutil.move(orig+ folder + '/images/'+image_folder[idx], dest+'/remains/images/'+image_folder[idx])
        shutil.move(orig+ folder + '/labels/'+label_folder[idx], dest+'/remains/labels/'+label_folder[idx])
        if image_folder[idx] != str(count_remain)+'.png':
            os.rename(dest+ '/remains/images/'+image_folder[idx],dest+'/remains/images/'+str(count_remain)+'.png')
            os.rename(dest+ '/remains/labels/'+label_folder[idx],dest+'/remains/labels/'+str(count_remain)+'.png')
        count_remain = count_remain + 1
    
        
    
    return (count_train , count_val , count_remain)


def filter_blank():
    image_folder = 'diverce dataset/final/town4/patch_dataset/validation/images'
    label_folder= 'diverce dataset/final/town4/patch_dataset/validation/labels'
    images = sorted_aphanumeric(os.listdir(image_folder))
    labels = sorted_aphanumeric(os.listdir(label_folder))
    for index in range(len(labels)):
        label = cv2.imread(os.path.join(label_folder,labels[index]),0)
        label = cv2.normalize(label.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        if label.sum() < 900:
            os.remove(os.path.join(label_folder,labels[index]))
            os.remove(os.path.join(image_folder,images[index]))
            


def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)






def Reorganize():
    path_data = 'diverce dataset/final/town7/patch_dataset'
    image_train_folder = path_data +'/train/images' 
    labels_train_folder = path_data +'/train/labels' 
    files_list = os.listdir(image_train_folder)
    labels_list = os.listdir(labels_train_folder)

    dest_data = 'E:/study/project_drone/Unet/finale_dataset/train/'
    dest_images = dest_data + '/images/'
    dest_labels = dest_data + '/labels/'

    
    
    count = 1937
    for ind in range(len(files_list)):
        # moves an image

        shutil.move(image_train_folder +'/'+ files_list[ind] , dest_images + str(count)+'.png' )
        shutil.move(labels_train_folder +'/'+ labels_list[ind] , dest_labels + str(count)+'.png' )
        count = count + 1
        
    



def MoveToDest_2(folder:str,ind_list:list,image_folder:list,label_folder:list,count:tuple = (0,0,0)):
    # move datasets after partition from orig to dest
    orig = "E:/study/project_drone/Unet/patch_dataset"
    dest = "E:/study/project_drone/Unet/new_patch_dataset"

    count_train , count_val , count_remain = count
    
    

    for idx in ind_list[0]:
        shutil.move(orig + folder + '/images/'+image_folder[idx], dest+'/train/images/'+image_folder[idx])
        shutil.move(orig+ folder + '/labels/'+ label_folder[idx], dest+'/train/labels/'+label_folder[idx])
        if image_folder[idx] !=str(count_train)+'.png':
            os.rename(dest+ '/train/images/' +image_folder[idx] , dest+'/train/images/'+str(count_train)+'.png')
            os.rename(dest+'/train/labels/'+label_folder[idx],dest+'/train/labels/'+str(count_train)+'.png')
        count_train = count_train + 1
        
        
    for idx in ind_list[1]:
        shutil.move(orig+ folder + '/images/'+image_folder[idx], dest+'/validation/images/'+image_folder[idx])
        shutil.move(orig+ folder + '/labels/'+label_folder[idx], dest+'/validation/labels/'+label_folder[idx])
        if image_folder[idx] != str(count_val)+'.png':
            os.rename(dest+ '/validation/images/'+image_folder[idx],dest+'/validation/images/'+str(count_val)+'.png')
            os.rename(dest+ '/validation/labels/'+label_folder[idx],dest+'/validation/labels/'+str(count_val)+'.png')
        count_val = count_val + 1
    
    for idx in ind_list[2]:
        shutil.move(orig+ folder + '/images/'+image_folder[idx], dest+'/remains/images/'+image_folder[idx])
        shutil.move(orig+ folder + '/labels/'+label_folder[idx], dest+'/remains/labels/'+label_folder[idx])
        if image_folder[idx] != str(count_remain)+'.png':
            os.rename(dest+ '/remains/images/'+image_folder[idx],dest+'/remains/images/'+str(count_remain)+'.png')
            os.rename(dest+ '/remains/labels/'+label_folder[idx],dest+'/remains/labels/'+str(count_remain)+'.png')
        count_remain = count_remain + 1
    
        
    
    return (count_train , count_val , count_remain)





def present_supervised_dataset(x,y):

    fig, ax = plt.subplots(nrows=2, ncols=3,figsize=(12,12))

    im_1 = x[0,:].clone().detach().numpy()*np.array([0.229, 0.224, 0.225]).reshape(-1,1,1) + np.array([0.485, 0.456, 0.406]).reshape(-1,1,1)
    #im_1 = im_1
    im_1 = np.transpose(im_1, axes=(1,2,0))
    ax[0][0].imshow(im_1)
    ax[0][0].set_title('image')

    im_2 = x[1,:].clone().detach().numpy()*np.array([0.229, 0.224, 0.225]).reshape(-1,1,1) + np.array([0.485, 0.456, 0.406]).reshape(-1,1,1)
    #im_2 = im_2
    im_2 = np.transpose(im_2, axes=(1,2,0))
    ax[0][1].imshow(im_2)
    ax[0][1].set_title('image')

    im_3 = x[2,:].clone().detach().numpy()*np.array([0.229, 0.224, 0.225]).reshape(-1,1,1)+ np.array([0.485, 0.456, 0.406]).reshape(-1,1,1)
    #im_3 = im_3
    im_3 = np.transpose(im_3, axes=(1,2,0))
    ax[0][2].imshow(im_3)
    ax[0][2].set_title('image')


    lb_1 = y[0,:].clone().detach().numpy()
    lb_1 = np.array([lb_1,lb_1,lb_1]).reshape(3,224,224)
    lb_1 = np.transpose(lb_1, axes=(1,2,0))
    ax[1][0].imshow(lb_1,cmap='gray')
    ax[1][0].set_title('label')

    lb_2 = y[1,:].clone().detach().numpy()
    lb_2 = np.array([lb_2,lb_2,lb_2]).reshape(3,224,224)
    lb_2 = np.transpose(lb_2, axes=(1,2,0))
    ax[1][1].imshow(lb_2,cmap='gray')
    ax[1][1].set_title('label')

    lb_3 = y[2,:].clone().detach().numpy()
    lb_3 = np.array([lb_3,lb_3,lb_3]).reshape(3,224,224)
    lb_3 = np.transpose(lb_3, axes=(1,2,0))
    ax[1][2].imshow(lb_3,cmap='gray')
    ax[1][2].set_title('label')




if __name__ == '__main__':
    #Reorganize()
    pass
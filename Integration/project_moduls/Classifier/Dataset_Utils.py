import glob
import re
import numpy as np
import shutil
import os 



def CreateDatasetFolder():
    count_empty = 0
    count_occ = 0
    data_set_folder = os.listdir('UFPR05')
     
    for weather_folder in data_set_folder:
        for date_folde in os.listdir('UFPR05/'+weather_folder+'/'):
            if len(os.listdir('UFPR05/'+weather_folder+'/')) == 0:
                continue
            for img in os.listdir('UFPR05/'+weather_folder+'/'+date_folde+'/Empty'):
                if len(os.listdir('UFPR05/'+weather_folder+'/'+date_folde+'/Occupied')) == 0:
                    continue
                shutil.move('UFPR05/'+weather_folder+'/'+date_folde+'/Empty'+'/'+img,'test/Empty/' +str(count_empty)+'.png')
                count_empty += 1

    for weather_folder in data_set_folder:
        for date_folde in os.listdir('UFPR05/'+weather_folder+'/'):
            if len(os.listdir('UFPR05/'+weather_folder+'/')) == 0:
                continue
            for img in os.listdir('UFPR05/'+weather_folder+'/'+date_folde+'/Occupied'):
                if len(os.listdir('UFPR05/'+weather_folder+'/'+date_folde+'/Occupied')) == 0:
                    continue
                shutil.move('UFPR05/'+weather_folder+'/'+date_folde+'/Occupied'+'/'+img,'test/Occupied/' +str(count_occ)+'.png')
                count_occ +=1



def TrimDataset():
    d_size = int(1e4)
    for folder in ['Empty','Occupied']:
        files_train = os.listdir(f'Parking/train/{folder}')
        ind_train = np.random.choice(len(files_train), d_size, replace=False)

        files_test = os.listdir(f'Parking/test/{folder}')
        ind_test = np.random.choice(len(files_test), d_size, replace=False)

        files_val = os.listdir(f'Parking/val/{folder}')
        ind_val = np.random.choice(len(files_val), d_size, replace=False)

        for ind in range(len(files_train)):
            if ind in ind_train:
                continue
            else:
                os.remove(f'Parking/train/{folder}/{files_train[ind]}')

        

        for ind in range(len(files_test)):
            if ind in ind_test:
                continue
            else:
                os.remove(f'Parking/test/{folder}/{files_test[ind]}')



        for ind in range(len(files_val)):
            if ind in ind_val:
                continue
            else:
                os.remove(f'Parking/val/{folder}/{files_val[ind]}')

        



    





if __name__ == '__main__':
    TrimDataset()


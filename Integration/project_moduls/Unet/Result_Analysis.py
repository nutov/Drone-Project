# -*- coding: utf-8 -*-
"""
Created on Tue May 12 12:51:21 2020

@author: notov
"""
import pandas as pd
import numpy as np
import torch
import csv

def WriteDictToCsv(csv_file:str,res:dict):
    '''
    saves the results to a csv file
    '''
    df = pd.DataFrame.from_dict(res, orient='columns')
    df.to_csv(csv_file)
    

def LoadCsvAndShowRes(file_name):
    df =  pd.read_csv(file_name)
    df.drop(df.columns[[0]], axis=1).plot()



def SaveModel(f_name:str,model):
    torch.save(model.state_dict(), f_name + '.pth')




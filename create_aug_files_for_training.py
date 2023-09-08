# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 10:06:14 2022

@author: Ibikunle Oluwanisola
"""
import os
import glob
from matplotlib import pyplot as plt
from scipy.io import loadmat,savemat
import numpy as np
from operator import itemgetter
from itertools import groupby

from albumentations import (
    Compose, RandomBrightness, RandomContrast, VerticalFlip,ElasticTransform,GaussianBlur,RandomBrightnessContrast, HorizontalFlip, ColorJitter
    
)


base_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\SR_Dataset_v1\Full'  # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'
train_path = os.path.join(base_path,'train_data\*.mat')


plot_figures = False
save_outputs = True        


all_transform = Compose( [RandomBrightness(limit=0.1),RandomBrightnessContrast(brightness_limit=0.10,contrast_limit=0.10, p= 0.3),VerticalFlip(), ElasticTransform(alpha=0.1,sigma = 0.2)])

transform1 = Compose([RandomBrightnessContrast(brightness_limit=0.15,contrast_limit=0.15, p= 0.3),  ElasticTransform(alpha= 2,sigma = 20)])
transform2 = Compose([RandomBrightnessContrast(brightness_limit=0.35,contrast_limit=0.35, p= 0.3)])

transform3 = Compose([ VerticalFlip(), ElasticTransform(alpha=0.1,sigma = 0.2) ])
transform4 = Compose([ GaussianBlur(), ColorJitter() ])

transform5 = Compose([ GaussianBlur(), HorizontalFlip() ])
transform6 = Compose([ RandomContrast(),ElasticTransform(alpha=0.05,sigma = 0.05) ])

def returnTransform(RandState):
    return {
         1: transform1, 2: transform2,
         3: transform3, 4: transform4,
         5: transform5, 6: transform6,
    }.get(RandState, transform1)

orig_train_data = glob.glob(train_path)

for elem in orig_train_data:
    
    orig_img = loadmat( elem )
    
    img = orig_img ['echo_tmp'].astype('float32')
    layer = orig_img['layers_segment_bitmap'] #semantic_seg
    
    attempts = 0
    
    for attempts in range(3):    
        try:
            RandState = np.random.randint(1,7)     
            transform = returnTransform(RandState) 
        except: 
            RandState = np.random.randint(1,7)     
            transform = returnTransform(RandState)             
        else:
            continue
        
    
    transformed = transform(image=img, mask=layer )
    
    echo_tmp = transformed["image"]
    semantic_seg2 = transformed["mask"].astype('float64')   
    
    # Ensure fixed Nt and Nx for echo and semantic seg
    Nt,Nx = echo_tmp.shape
    semantic_seg2 = semantic_seg2[:Nt,:Nx]
    
    
    # Create new raster
    raster = np.zeros_like(echo_tmp)    
    for col_idx in range(echo_tmp.shape[1]):
         
         repeat_tuple = [ (k,sum(1 for _ in groups)) for k,groups in groupby(semantic_seg2[:,col_idx]) ]            
         raster_val_idx = np.cumsum([ item[1] for item in repeat_tuple]) # Cumulate the returned index
         raster_val_idx = raster_val_idx[:-1]
         
         # _, raster_val_idx = np.unique(semantic_seg2[:,col_idx],return_index=True)
         # raster_val_idx = raster_val_idx[raster_val_idx > 0]  
         
         raster[ np.array(raster_val_idx), col_idx ] = 1
         
         
    if plot_figures:
        f, axarr = plt.subplots(1,5,figsize=(20,20))

        axarr[0].imshow(img.squeeze(),cmap='gray_r');
        axarr[0].set_title(f'{RandState}')
        axarr[1].imshow(layer, cmap='viridis' )
        axarr[2].imshow(echo_tmp, cmap='gray_r' )
        axarr[3].imshow(semantic_seg2, cmap='viridis') 
        axarr[4].imshow(raster, cmap='gray_r')
    
    if save_outputs:
        
        ## Save outputs ( augmented png and .mat file)
        
        # 1. Save .png        
        aug_file_path = os.path.join( os.path.dirname(os.path.dirname(elem) ), 'augmented_data')
        
        if not os.path.exists(aug_file_path):
            os.mkdir(aug_file_path)
        
        full_path_png = os.path.join(aug_file_path, os.path.splitext(os.path.basename(elem))[0]+'.png'  )
        plt.imsave(full_path_png, echo_tmp, cmap= 'gray')
        
        # 2. Save .mat file
        
        # Save original values 
        out_dict = {key: orig_img.get(key) for key in ['layers_vector', 'Elevation', 'Latitude',  'Longitude', 'original_frame']} #,'new_Elev', 'new_lat','new_lon','original_frame', vec_layer
        
        # Add new echo, raster and seg
        out_dict['semantic_seg2'] = semantic_seg2
        out_dict['echo_tmp'] = echo_tmp
        out_dict['raster'] = raster
        
        # Save
        full_path_mat = os.path.join(aug_file_path, os.path.splitext(os.path.basename(elem))[0]+'_aug.mat')
        savemat(full_path_mat,out_dict)
        
        
    
   
    
    
    
    
    
    
    
    
    
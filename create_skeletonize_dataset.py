# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 00:13:43 2023

Create Skeletonize dataset

@author: i368o351
"""

import os
import cv2
import glob
from scipy.io import loadmat,savemat
import numpy as np

base_path = r'X:\public\data\temp\internal_layers\NASA_OIB_test_files\image_files'
out_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\skeletonize'

d1 = r'high_variance_flat_layers\image\*.mat'
d2 = r'high_variance_non_flat_layers\image\*.mat'
d3 = r'new_high_variance_non_flat_layers\image\*.mat'

d1_path = os.path.join(base_path,d1)
d2_path = os.path.join(base_path,d2)
d3_path = os.path.join(base_path,d3)


d1_files = glob.glob(d1_path)
d2_files = glob.glob(d2_path)
d3_files = glob.glob(d3_path)

Ds = ['d1','d2','d3']

all_files = d1_files + d2_files + d3_files

Nt,Nx = 1664,256

for f_idx,folder in enumerate([d1_files, d2_files, d3_files]):
    
    for file in folder: 
        
        curr_file = loadmat(file)['layer']
        base_name =  os.path.basename(file)    
        
        raster = np.zeros(shape=(Nt,Nx))
        new_layer = np.zeros(shape=(Nt,Nx))
        
        n_rows = curr_file.shape[0]
        
        # Randomize
        num_layers_w_gap = np.random.randint(0,n_rows//4)
        layers_w_gap = np.random.randint(0,n_rows,size=(num_layers_w_gap,) )    
        iterations = np.random.randint(1,3)
        k1 = np.random.randint(5,21)
        k2 = np.random.randint(11,61)
        
        kernel = np.ones((k1,k2))
        
        for rw_idx in range(curr_file.shape[0]):
            
            curr_row =  curr_file[rw_idx]
            cols_to_use = list(range(Nx))
            
            raster[curr_row,cols_to_use] = 1 # Perfect raster
            
            # Create dilated raster        
            if layers_w_gap.ndim and layers_w_gap.size and rw_idx in layers_w_gap:
                gap_start =  np.random.choice(list(range(Nx))) 
                gap_length = np.random.randint(10,Nx//2)
                
                gap = list(range(gap_start, max(Nx, gap_start+gap_length) ))
                cols_to_use = list(set(range(Nx)) - set(gap) )
                
                curr_row = curr_row[cols_to_use]
                
                
            new_layer [curr_row, cols_to_use ] = 1
            
        # Dilate layers
        new_layer = cv2.dilate(new_layer, kernel, iterations = iterations)
        
        res = {}
        res['dil_raster'] = new_layer
        res['raster'] = raster
        res['vec_layer'] = curr_file
        
        # Save file
        if not os.path.exists(os.path.join(out_path)):
            os.makedirs(os.path.join(out_path), exist_ok=True) 
        
        l_fname = base_name [:-4]+ f'_{Ds[f_idx]}'+ base_name[-4:]
        
        curr_out_path = os.path.join(out_path,l_fname)        
        savemat(curr_out_path, res)
      
      
    
    









# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 11:52:56 2023
 Script sort_small_list for make_vec_layer
@author: i368o351
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def make_vec_layer_old(raster):
    # Only works for perfect situations
    Nt,Nx = raster.shape
    temp1 = np.argwhere(raster)
    bin_rows,bin_cols = temp1[:,0], temp1[:,1]
    
    
    col_list = []
    
    orig_col_list = []
    orig_row_list = []
    
    final_col_list = []
    final_row_list = []
    
    idx_list = []
    
    for idx,elem in enumerate(bin_cols):    
        
        if (elem not in col_list) and (np.abs(bin_rows[idx] -bin_rows[idx]) < 5) :        
            col_list.append(elem)
            idx_list.append(idx)
        else:
            
            orig_col_list.append(col_list)
            orig_row_list.append(bin_rows[idx_list])
            
            col_list = sorted(col_list)
            idx_list = sorted(idx_list)
            
            final_col_list.append(col_list)
            final_row_list.append(bin_rows[idx_list])
            
            idx_list = [idx]
            col_list = [elem]
            
    layers = np.empty( (len(final_col_list ), Nt))   
    layers[:] = np.nan
    
    for iter in range(len(final_col_list)):
        layers[ iter, final_col_list[iter] ] = final_row_list[iter]
            
    return layers

aa = loadmat('Y:/ibikunle/Python_Project/Fall_2021/Predictions_Folder/EchoViT_out/L2/col_embed_out/20120330_04_0232_5km.mat')
raster = aa['binary_output']

# def make_vec_layer3(raster):
    
Nt,Nx = raster.shape
temp1 = raster.T.nonzero()
bin_cols,bin_rows = temp1[0],temp1[1]

# Remove infrequent rows
bins, bin_idx, bin_cnt = np.unique(bin_rows, return_index=True, return_counts=True)

uniq_cols = np.unique(bin_cols)

layer1 = []
for col in uniq_cols:
    layer1.append(bin_rows [bin_cols == col])

col_lengths = [len(item) for item in layer1]
max_len, max_idx = np.max(col_lengths), np.argmax(col_lengths)

vec_layer = np.full( (max_len,Nx), fill_value = np.nan)

short_col_idx = [ii for ii,xx in enumerate(col_lengths) if xx < max_len ]

max_col = np.array(layer1[max_idx])

threshs = np.round( 3* np.exp(0.3*np.linspace(0,20,20)) )

for col_idx in range(Nx):
    curr_short_col = np.array(layer1[col_idx] )
    
    if col_idx in short_col_idx:
        mult = len(max_col) - len(curr_short_col) 
        fix_pos_list = []
        # for repeat in range(mult):
        ind = abs(max_col - curr_short_col[:,None]) <= 5 #threshs[repeat] 
        if np.all(np.diag(ind)):
            ind = abs(max_col - curr_short_col[:,None]) <= threshs[0] # Use smallest thresh
            
        fix_pos = np.where(~(ind.max(0)))[0][0]
        fix_pos_list.append(fix_pos)
        fix_val = max_col [fix_pos] 
        # Need to fix case where multiple nans need to be inserted
        curr_short_col = np.insert( curr_short_col, fix_pos, fix_val )
    
    curr_short_col[fix_pos] = 0
    vec_layer[:,col_idx]  = curr_short_col
    vec_layer [vec_layer == 0 ] = np.nan
    
    _ = plt.plot(vec_layer.T); _ = plt.gca().invert_yaxis()
    
    
    
    
    
    
    
    
    
    
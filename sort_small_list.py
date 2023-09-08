# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 11:52:56 2023
 Script sort_small_list for make_vec_layer
@author: i368o351
"""

import numpy as np

def sort_small_list(raster):
    temp1 = np.argwhere(raster)
    bin_rows,bin_cols = temp1[:,0], temp1[:,1]
    
    
    col_list = []
    
    final_col_list = []
    final_row_list = []
    
    idx_list = []
    
    for idx,elem in enumerate(bin_cols):    
        
        if elem not in col_list:        
            col_list.append(elem)
            idx_list.append(idx)
        else:
            col_list = sorted(col_list)
            idx_list = sorted(idx_list)
            
            final_col_list.append(col_list)
            final_row_list.append(bin_rows[idx_list])
            
            idx_list = [idx]
            col_list = [elem]
    return final_row_list,final_col_list
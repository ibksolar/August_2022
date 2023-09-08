# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:17:26 2023

@author: i368o351
"""

import numpy as np

import collections
def scan_count(l):
    count = collections.defaultdict(int)
    for i in l:
        yield count[i]
        count[i] += 1
        
def scan_count2(l):
    final_list = []
    count = collections.defaultdict(int)
    for i in l:
        final_list.append(count[i])
        count[i] += 1  
    
    return final_list

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



def make_vec_layerv2(raster,threshold= 10):
    '''              
    Parameters
    ----------
    raster : TYPE: numpy array
        DESCRIPTION: raster vector with lots of zeros (size Nx * Nt) 
                     It's typically map of 0's and 1's that has been multiplied with rangebin index'
                     
    threshold : TYPE: int (scalar): this should eventually be estimated from data
        DESCRIPTION: Determines the minimum jump between 2 consecutive layers        


    Returns
    -------
    layers_new : TYPE: numpy array
        DESCRIPTION: vectorized layers with zeros removed (size: Num_layers x Nt)
    
    Example usage:
        layers_new = make_vec_layerv2(res0_final, 10)

    '''
    
    # (1) Remove short rows
    
    diff_temp = np.argwhere(raster) #raster.nonzero()
    Nx = raster.shape[-1]
    
    # diff_tuple = [ (diff_temp[iter,0],diff_temp[iter,1]) for iter in range(diff_temp.shape[0]) ]    
    # diff_temp =  np.asarray( sorted(diff_tuple, key=lambda x: (x[0],x[1])) )
    
    bin_rows,bin_cols = diff_temp[:,0], diff_temp[:,1]    
    bin_rows +=1 # Correct offset
    
    all_rows_to_rmv = []   

    
    ## Find short rows and remove them   
    rw_uniq,rw_idx, rw_inv, rw_count = np.unique(bin_rows,return_index=True,return_counts=True,return_inverse=True)
    
    short_rows= rw_uniq [rw_count <= Nx//6 ] 
    # short_cols = bin_cols [ rw_idx [rw_count <= Nx//6 ] ]
    
    for curr_short_rw in short_rows:
       
        curr_neigh = rw_uniq[ (np.abs(curr_short_rw - rw_uniq)) <=15 ]  # curr_neigh = rw_uniq[ (np.abs(curr_short_rw - rw_uniq)).argsort()[:5] ]
        
        if np.any(curr_neigh):
            # curr_neigh = curr_neigh [np.abs( np.array(curr_short_rw) - curr_neigh) <= 20 ] # Neighboring rows mst not be farther than 20
            sum_neighbors = sum( [ sum(bin_rows==elem) for elem in curr_neigh])
            
            if sum_neighbors < Nx//5:
                raster[raster==curr_short_rw] = 0
    
    
    diff_temp = np.argwhere(raster)    
    bin_rows,bin_cols = diff_temp[:,0], diff_temp[:,1]    
    
    # Correct offsets (first and second derivative)
    bin_rows +=1 # Need to confirm this
    
    dy2dx = np.append( np.array([0]), np.diff(np.diff(bin_rows)))
    loc_idx = np.where(dy2dx <-threshold)[0]
    loc_idx = np.concatenate( (np.array([0]),loc_idx,np.array(len(bin_rows)-1)), axis= None)
    
    rows_new, cols_new = [],[]
    
    for ix in range(len(loc_idx)-1):
        
        curr_loc = loc_idx[ix]        
        next_loc = loc_idx[ix+1] if ix < len(loc_idx)-1 else loc_idx[-1]
        
        if len( range(curr_loc,next_loc) ) <= Nx:
            rows_new.append( np.array(bin_rows[curr_loc:next_loc]) )
            cols_new.append( np.array(bin_cols[curr_loc:next_loc]) )
        else:
            # Peaks lesser than threshold or towards the end
            while curr_loc +1 < next_loc:
                
                max_val = np.min((dy2dx[curr_loc:  np.min((curr_loc+Nx+1,loc_idx[-1]-1))  ] ))
                max_loc = np.where((dy2dx[curr_loc:  np.min((curr_loc+Nx+1,loc_idx[-1]-1))  ] ) == max_val  )[0] 
                
                max_loc = max_loc if len(max_loc) < 2 else max_loc[-1] # Find last occurrence
                max_loc = max_loc.item() # Max loc should be the beginning of the next layer               
                
                if np.abs(max_val) < 5 : #or max_loc < 5
                    # Let's avoid deleting: Sort and use columns
                    # Check columns
                    tmp_cols = bin_cols[curr_loc:  np.min((curr_loc+Nx+1,loc_idx[-1]-1))  ]
                    srtd_tmp_cols, idx_list =  [], []
                    for idx,elem in enumerate(tmp_cols):
                        while elem not in srtd_tmp_cols:
                            srtd_tmp_cols.append(elem)
                            idx_list.append(idx)
                    max_loc =  idx_list[-1]
                        
                if curr_loc + Nx < loc_idx[-1]:            
                    rows_new.append( np.array(bin_rows[curr_loc:curr_loc+max_loc]) ) # Max loc should be not be included in the current layer  
                    cols_new.append( np.array(bin_cols[curr_loc:curr_loc+max_loc]) )                 
                        
                    curr_loc += max_loc                      
                else:
                    
                    rows_new.append( np.array(bin_rows[curr_loc:loc_idx[-1]] ) )
                    cols_new.append( np.array(bin_cols[curr_loc:loc_idx[-1]] ) )
                    
                    curr_loc = loc_idx[-1]
    
          
    layers_new = np.full( ( len(cols_new) ,Nx), np.nan)
    
    for iter in range(len(cols_new) ):
        layers_new[iter,cols_new[iter]] = rows_new[iter]
        
    return layers_new



        
        
        
        
        
        
        
        
        
        
        
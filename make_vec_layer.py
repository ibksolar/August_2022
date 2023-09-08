# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:35:40 2022

@author: i368o351
"""

import numpy as np

def make_vec_layer(raster,threshold={'constant':25}, debug = False):
    '''              
    Parameters
    ----------
    raster : TYPE: numpy array
        DESCRIPTION: raster vector with lots of zeros (size Nx * Nt) 
                     It's typically map of 0's and 1's that has been multiplied with rangebin index'
    threshold : TYPE: int (scalar)
        DESCRIPTION: Determines the minimum jump between 2 consecutive layers
    debug: TYPE: boolean
        DESCRIPTION: True --> Show inner workings and print intermediate outputs. False --> ignores prints

    Returns
    -------
    vec_layer : TYPE: numpy array
        DESCRIPTION: vectorized layers with zeros removed (size: Num_layers x Nt)
    
    Example usage:
        vec_layer = create_vec_layer(res0_final, 10)

    '''
    
    #TO DO: Check type of raster; should be numpy array
    
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
                rows_to_rmv = np.argwhere(bin_rows==curr_short_rw)
                all_rows_to_rmv.append( [rm_item.item() if rm_item <len(bin_rows) else rm_item.item()-1 for rm_item in rows_to_rmv] )
    
    if all_rows_to_rmv:
        all_rows_to_rmv = [elem2 for elem1 in all_rows_to_rmv for elem2 in elem1 ]
        diff_temp = np.delete(diff_temp, all_rows_to_rmv, axis = 0)             
        
    bin_rows,bin_cols = diff_temp[:,0], diff_temp[:,1]    
    bin_rows +=1 # Correct offset
    
    
    if 'constant' in threshold.keys():
        threshold = threshold['constant'] * np.ones(shape=(10000,))
    else:
        threshold = np.round( list( threshold.values())[0] *np.exp(0.008*np.linspace(1,100,100)) )
           
    #threshold = np.round( threshold*np.exp(0.008*np.linspace(1,100,100)) )
    
    
    # Initialize
    brk_points = [ 0 ]    
  
    vec_layer = np.zeros( shape=(1,Nx) ) # Total number of layers is not known ahead of time
    
    # Initializations
    brk_pt_start = 0;  brk_pt_stop = Nx+5; 
    
    count = 0
    
    while brk_pt_start < len(bin_rows):
        # if ( np.diff(bin_rows[brk_pt_start:brk_pt_stop] ) >= min_jump ).any():
        if brk_pt_stop < len(bin_rows):
            
            # curr_cols_to_use = bin_cols [brk_pt_start:brk_pt_stop]
            # curr_rows_to_use = bin_rows[brk_pt_start:brk_pt_stop]
            
            # first_uniq, first_uniq_idx = np.unique(curr_cols_to_use, return_index=True)
            # remaining_idx = np.delete( np.arange(len(curr_cols_to_use)),first_uniq_idx)
            
            # rem_cols = curr_cols_to_use[remaining_idx]
            # remaining_idx = np.argsort(rem_cols)
            
            # final_sorted_idx = np.concatenate( (first_uniq_idx,remaining_idx), axis=0)
            
            # sorted_curr_rows_to_use = curr_rows_to_use[final_sorted_idx]
            
            
            curr_max_jump = np.max( np.diff (bin_rows[brk_pt_start:brk_pt_stop]) )            
            curr_max_loc = np.where(np.diff(bin_rows[brk_pt_start:brk_pt_stop]) >= curr_max_jump )[0]
            
            curr_max_loc = np.array([ np.min((Nx - 1,curr_max_loc[-1])) ]) if len(curr_max_loc)>1 else curr_max_loc
            
            if curr_max_loc  > int(.9*Nx):                
                tmp_res = curr_max_loc
                
            else:                                
                # If less than: there's a break or the current layer is not the entire Nx long.
                # Check if the current jump does not exceed max allowed(i.e threshold). If it does; split into a new layer else attempt to use all Nx-1
                if curr_max_jump > threshold[count]:
                    tmp_res =  curr_max_loc                
                else:
                    # Search further ignoring the current max jump since it's still lower than global threshold
                    
                    # (a.) Check if there's a full layer directly beneath
                    # If yes, truncate early
                    if len( bin_rows[brk_pt_start+ curr_max_loc.item() +1 : min(brk_pt_start+ curr_max_loc.item() + Nx+5,len(bin_rows) )  ]) > 1:                    
                        further_search = np.argmax ( np.diff(bin_rows[brk_pt_start+ curr_max_loc.item() +1 : min(brk_pt_start+ curr_max_loc.item() + Nx+5,len(bin_rows) )  ] ) )
                        if  further_search !=0 and further_search >= int(.6*Nx):
                            
                            if curr_max_loc.item() <= Nx//5: # Need to debug this
                                brk_pt_start = brk_pt_start + curr_max_loc.item() + 1
                                brk_pt_stop = brk_pt_start + Nx + 5 
                                continue
                            else:
                                tmp_res = curr_max_loc
                        else: # If no, continue search for the break point
                            next_max = np.max ( np.diff(bin_rows[brk_pt_start+ curr_max_loc.item() +1 :brk_pt_stop]) )
                            tmp_res = curr_max_loc +  np.where(np.diff(bin_rows[brk_pt_start+curr_max_loc.item() :brk_pt_stop]) == next_max) [0]
                    else:
                        tmp_res = curr_max_loc
                
            if len(tmp_res)>1:
                brk_pt_stop = (tmp_res[tmp_res>0][0] + brk_pt_start).item()
            else:
                brk_pt_stop = (tmp_res  + brk_pt_start ).item()
        
        
        else: # Consecutive layers with very small breaks between them
            brk_pt_stop = len(bin_rows)
            
            # if brk_pt_stop < len(bin_rows):                  
            #     tmp_res = np.array([Nx - 1])  #np.argmax(np.diff (bin_rows[brk_pt_start:brk_pt_stop]) ) #np.array([Nx - 1]) 
            #     brk_pt_stop = (tmp_res  + brk_pt_start ).item()                
            # else: # Should be the last layer
                                               
            
        vec_layer = np.concatenate( (vec_layer,np.zeros(shape=(1,Nx)) ) )
        used_cols = bin_cols[brk_pt_start:brk_pt_stop+1] # Added extra 1 because of zero indexing
        used_rows = bin_rows[brk_pt_start:brk_pt_stop+1 ] # Added extra 1 because of zero indexing
        
        
        # This stage might not be needed after sorting diff temp implementation
        _,used_cols_unq = np.unique(used_cols,return_index=True)
        
        used_cols = used_cols[used_cols_unq]
        used_rows = used_rows[used_cols_unq]
         
        vec_layer[-1, used_cols ] = used_rows                                   
        brk_points.append( (brk_pt_start,brk_pt_stop,brk_pt_stop-brk_pt_start, list(used_rows) ) )        
                                             
        
        if debug:
            # To be removed after debugging
            brk_pt_stop = brk_pt_stop-2 if brk_pt_stop >= len(bin_rows) else brk_pt_stop         
            print('=================================================================================================')
            print(f'Current layer have {brk_pt_stop - brk_pt_start} elements')
            print(f'Layer {count}: Ends with { bin_rows[brk_pt_stop]} ({brk_pt_stop}), next layer starts with {bin_rows[brk_pt_stop+1]} {(brk_pt_stop+1)}')
            print(f'layer jump is {bin_rows[brk_pt_stop+1]} - {bin_rows[brk_pt_stop] } = {bin_rows[brk_pt_stop+1] - bin_rows[brk_pt_stop] } ')
            print('=================================================================================================')
            
        
        brk_pt_start = brk_pt_stop + 1
        brk_pt_stop = brk_pt_start + Nx + 5 # Adding extra one to complete Nx ( realizing Python indexing w/o last element)
        
        count +=1

    return vec_layer






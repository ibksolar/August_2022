# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 22:26:21 2023

@author: i368o351
"""

import numpy as np

def create_vec_layer(raster,threshold= 12):
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
    layers_new : TYPE: numpy array
        DESCRIPTION: vectorized layers with zeros removed (size: Num_layers x Nt)
    
    Example usage:
        layers_new = create_vec_layer(res0_final, 10)
'''



    diff_temp = np.argwhere(raster) #raster.nonzero()        
    Nx = raster.shape[-1]   
   
    bin_rows,bin_cols = diff_temp[:,0], diff_temp[:,1]    
    
    if 1:
        ## Find short rows and remove them   
        rw_uniq,rw_idx, rw_inv, rw_count = np.unique(bin_rows,return_index=True,return_counts=True,return_inverse=True)        
        short_rows= rw_uniq [rw_count <= Nx//7 ]    
        
        for curr_short_rw in short_rows:       
            curr_neigh = rw_uniq[ (np.abs(curr_short_rw - rw_uniq)) <=15 ]  # curr_neigh = rw_uniq[ (np.abs(curr_short_rw - rw_uniq)).argsort()[:5] ]
            
            if np.any(curr_neigh):
                # curr_neigh = curr_neigh [np.abs( np.array(curr_short_rw) - curr_neigh) <= 20 ] # Neighboring rows must not be farther than 20
                sum_neighbors = sum( [ sum(bin_rows==elem) for elem in curr_neigh])        
                if (curr_short_rw+sum_neighbors) < Nx//5:
                    raster[raster==curr_short_rw] = 0
    
    
    diff_temp = np.argwhere(raster) 
    diff_temp =  np.asarray( sorted(diff_temp, key=lambda x: (x[0],-x[1])) )
    diff_temp[:,0] +=1 # Correct offsets 
    bin_rows,bin_cols = diff_temp[:,0], diff_temp[:,1]   
    bin_rows = np.array(bin_rows).astype('float32')
    
    dy2dx = np.append( np.array([0]), np.diff(np.diff(bin_rows))) # Add leading zero
    dy2dx = np.append( dy2dx,np.array([0]) )  # Add trailing zero
    
    loc_idx = np.where(dy2dx <-threshold)[0]
    loc_idx = np.concatenate( (np.array([0]),loc_idx,np.array(len(bin_rows)-1)), axis= None)
    
    rows_new, cols_new = [],[]
    
    for ix in range(len(loc_idx)-1):
        
        curr_loc = loc_idx[ix]        
        next_loc = loc_idx[ix+1] if ix < len(loc_idx)-1 else loc_idx[-1]
        
        if len( range(curr_loc,next_loc) ) <= Nx:
            rows_new.append( np.array(bin_rows[curr_loc:next_loc]) )
            cols_new.append( np.array(bin_cols[curr_loc:next_loc]) )
            
            # Set all used bin_rows to NaN
            bin_rows[curr_loc:next_loc] = np.nan
        
        elif curr_loc > loc_idx[-2] and curr_loc < loc_idx[-1]:
            rows_new.append( np.array(bin_rows[curr_loc:loc_idx[-1] ]) )
            cols_new.append( np.array(bin_cols[curr_loc:loc_idx[-1] ]) )
            
        else:
            # Peaks lesser than threshold or towards the end
            while curr_loc + 1 < next_loc:
                
                remaining = np.where(~np.isnan(bin_rows))[0]
                
                if remaining.size and remaining.ndim:
                    stop_srch = next_loc - remaining[0] ; 
                    search_idx = remaining[ :min(Nx,stop_srch) ] if len(remaining) >=Nx-1 else remaining[:]  
                
                    if search_idx.size and search_idx.ndim:                        
                        max_val = np.max((dy2dx[search_idx ] ))
                        max_loc = np.where((dy2dx[search_idx] ) == max_val  )[0] 
                        
                        max_loc = max_loc if len(max_loc) < 2 else max_loc[-1] # Find last occurrence
                        max_loc = max_loc.item() # Max loc should be the beginning of the next layer               
                                            
                        tent_cols = bin_cols[ search_idx ]                    
                        first_chk = np.mean( bin_rows[ search_idx[np.argsort(tent_cols)[:10]]] )                    
                        
                        used_rw_idx = search_idx[:max_loc+1] # This may be overwritten if np.abs(max_val) < 5 is True                   
                        
                        if (max_loc < Nx or np.abs(max_val) <= 3) and len(remaining) >=Nx-1 : # (max_loc+next_max_loc) <= Nx: #
                        # if max_loc < Nx-1: #or max_loc < 5 #np.abs(max_val) < 10 and
                            # Sort and use columns                      
                            search_idx = remaining[ :round(3*Nx) ] if len(remaining) >=Nx-1 else remaining[:] # Expand search
                            tmp_cols = bin_cols[search_idx]                        
                            srtd_tmp_cols, idx_list =  [], []
                    
                            # Get all next >Nx columns
                            # What if earlier cols e.g 0-10 actually belong to the next(lower) layer?
                            for iter in range(Nx): 
                                idx = np.argwhere(tmp_cols == iter )
                                if idx.ndim and idx.size : #or np.any(idx) or idx.item() == 0
                                    idx = (idx.T)[0]                               
                                    decider = first_chk if not idx_list else bin_rows[search_idx[idx_list[-1]]]
                                    
                                    # If there are multiple candidates   (#idx = idx.item() if len(idx) < 2 else [ idx_elem.item() for idx_elem in idx])
                                    # Choose the one closest to the previous column
                                    if len(np.asarray(idx)) > 1:
                                        pot_bin_rws = [bin_rows[search_idx[elem.item()]] for elem in idx] # bin rows of potential candidates
                                        idx = idx[np.argmin(abs(decider-pot_bin_rws))].item() # use the closest bin_rows
                                    else:
                                        idx = idx.item()                                   
          
                                    # Create Flag to check if there are middle layers between decider (last col bin_rows) and the next one to be decided
                                    # new_bin_rw_b4 = [elem for elem in bin_rw_b4 if elem > decider]
                                                                   
                                    
                                    # Check if it's not greater than threshold or if the column before does not have a valid neighbor
                                    if np.abs( bin_rows[search_idx[idx]] - decider ) <= threshold:
                                        idx_list.append(idx) # idx_list is the index of bin rows/cols
                                        srtd_tmp_cols.append(iter) # Might not be necessary
                                    else:
                                        # start_pt = max(0, len(srtd_tmp_cols)- 10 )
                                        # iter_list = srtd_tmp_cols[start_pt:] if len(srtd_tmp_cols)>0 else [iter + 2] 
                                        iter_list = list(range( max(0,iter-20),min(iter+20,Nx) )) if iter>0 else [2]  # range(max(1,iter-30),iter)     
                                        
                                        if iter_list:
                                            for ii,curr_iter in enumerate(iter_list):
                                                if ii == 0:
                                                    bin_rw_b4 = np.ma.array( bin_rows[bin_cols==curr_iter-1], mask=np.isnan(bin_rows[bin_cols==curr_iter-1]))
                                                else:
                                                    t1 = np.ma.array( bin_rows[bin_cols==curr_iter-1], mask=np.isnan(bin_rows[bin_cols==curr_iter-1]))
                                                    bin_rw_b4 = np.append(bin_rw_b4,t1)
                                       
                                        # Second opportunity to add 
                                        # TO DO: ( Upgrade this to handle more difficult case)
                                        if not bin_rw_b4.mask.all() and ~np.ma.any( np.abs(bin_rows[search_idx[idx]] - bin_rw_b4) <=5 ) : 
                                            idx_list.append(idx) # idx_list is the index of bin rows/cols
                                            srtd_tmp_cols.append(iter) # Might not be necessary                                             
                                                    
                            # idx_list is the index of bin rows/cols
                            used_rw_idx = sorted(search_idx[idx_list])                    
        
                           
                        if curr_loc + Nx < loc_idx[-1]:            
                            rows_new.append( np.array(bin_rows[used_rw_idx]) ) # Max loc should be not be included in the current layer  
                            cols_new.append( np.array(bin_cols[used_rw_idx]) )                 
                                
                            # curr_loc += max_loc                         
                        else:                        
                            rows_new.append( np.array(bin_rows[curr_loc:loc_idx[-1]] ) )
                            cols_new.append( np.array(bin_cols[curr_loc:loc_idx[-1]] ) )                   
                        
                        # Set all used bin_rows to NaN
                        bin_rows[used_rw_idx] = np.nan
                        
                    
                # Find the next start
                new_rem = np.argwhere(~np.isnan(bin_rows))
                
                # Remove artifacts from earlier layers and set to nan
                if new_rem.size and np.any(abs(np.diff(new_rem.ravel())) > 100 ):
                    new_start_idx = np.argwhere( abs(np.diff(new_rem.ravel() )) > 100 )[0]
                    new_start_idx = new_start_idx.item() if len(new_start_idx) < 2 else new_start_idx[-1].item()
                    bin_rows[new_rem[:new_start_idx + 1]] = np.nan
                    new_rem = np.argwhere(~np.isnan(bin_rows))            
                
                curr_loc = new_rem[0].item() if new_rem.size and new_rem.ndim else loc_idx[-1]    
                
          
    layers_new = np.full( ( len(cols_new) ,Nx), np.nan)
    
    for iter in range(len(cols_new) ):
        layers_new[iter,cols_new[iter]] = rows_new[iter]
        
    return layers_new

















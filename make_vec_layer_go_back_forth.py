# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 12:36:35 2023

@author: i368o351
"""

# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy.io import loadmat
import scipy

import glob
import matplotlib.pyplot as plt

import matplotlib as mpl

from scipy.ndimage import median_filter as sc_med_filt
from matplotlib import colors

import collections

# Set the default color cycle
ab = list( mpl.cycler(mpl.rcParams['axes.prop_cycle']) )
clr = [ item['color'] for item in ab if item['color'] != '#7f7f7f' ] # Remove illegible color
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=clr) 


base_path = r'Y:/ibikunle/Python_Project/Fall_2021/Predictions_Folder/EchoViT_out/all_files'
echo_path = r'Y:/ibikunle/Python_Project/Fall_2021/all_block_data/Attention_Train_data/Full_size_data/test_data'

def scan_count2(l):
    final_list = []
    count = collections.defaultdict(int)
    for i in l:
        final_list.append(count[i])
        count[i] += 1      
    return final_list

all_files = glob.glob(os.path.join(base_path,'*.mat'))

for file_idx, file in enumerate(all_files):
    base_name = os.path.basename(file)
    
    
    aa = loadmat(file)     
    raster = aa['binary_output']
    debug_plot = 0
    
    if np.sum(raster)>1 and base_name in ['20120330_04_1205_2km.mat']: #base_name in '20120330_04_1205_2km.mat' ['20120330_04_0220_5km.mat']: # np.sum(raster) > 0 and  base_name in ['20120330_04_0207_5km.mat']: #'20120330_04_0220_5km.mat',, '20120330_04_0208_5km.mat', 20120330_04_0207_5km.mat,base_name in ['20120330_04_0207_5km.mat'] and
        print(f'Now processing {base_name} ....{file_idx} of {len(all_files)}')
        echo_img = loadmat(os.path.join(echo_path,base_name))['echo_tmp']
        Nx = echo_img.shape[1]
        
        # vec_layer = make_vec_layerv2(raster,threshold=15)
        threshold = 12
        diff_temp = np.argwhere(raster) #raster.nonzero()        
        Nx = raster.shape[-1]
        
        # diff_tuple = [ (diff_temp[iter,0],diff_temp[iter,1]) for iter in range(diff_temp.shape[0]) ]    
        # diff_temp =  np.asarray( sorted(diff_tuple, key=lambda x: (x[0],x[1])) )       
        
        bin_rows,bin_cols = diff_temp[:,0], diff_temp[:,1]    
        all_rows_to_rmv = []   

        
        ## Find short rows and remove them   
        rw_uniq,rw_idx, rw_inv, rw_count = np.unique(bin_rows,return_index=True,return_counts=True,return_inverse=True)        
        short_rows= rw_uniq [rw_count <= Nx//6 ] 
        # short_cols = bin_cols [ rw_idx [rw_count <= Nx//6 ] ]
        
        
        if debug_plot:
            plt.figure(2)
            _ = plt.imshow(raster)
        
        for curr_short_rw in short_rows:
           
            curr_neigh = rw_uniq[ (np.abs(curr_short_rw - rw_uniq)) <=15 ]  # curr_neigh = rw_uniq[ (np.abs(curr_short_rw - rw_uniq)).argsort()[:5] ]
            
            if np.any(curr_neigh):
                # curr_neigh = curr_neigh [np.abs( np.array(curr_short_rw) - curr_neigh) <= 20 ] # Neighboring rows mst not be farther than 20
                sum_neighbors = sum( [ sum(bin_rows==elem) for elem in curr_neigh])
                
                if sum_neighbors < Nx//4:
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
                        
                        # next_max = np.max( dy2dx[max_loc+1: max_loc+Nx+1] )
                        # next_max_loc  = np.where(dy2dx[max_loc+1: max_loc+Nx+1]  == next_max  )[0] 
                        # next_max_loc = next_max_loc.item() if len(next_max_loc) < 2 else next_max_loc[-1].item()
                        
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
                                        iter_list = list(range(iter)) if iter>0 else [2]  # range(max(1,iter-30),iter)                                    
                                        if iter_list:
                                            for ii,curr_iter in enumerate(iter_list):
                                                if ii == 0:
                                                    bin_rw_b4 = np.ma.array( bin_rows[bin_cols==curr_iter-1], mask=np.isnan(bin_rows[bin_cols==curr_iter-1]))
                                                else:
                                                    t1 = np.ma.array( bin_rows[bin_cols==curr_iter-1], mask=np.isnan(bin_rows[bin_cols==curr_iter-1]))
                                                    bin_rw_b4 = np.append(bin_rw_b4,t1)
                                       
                                        # Second opportunity to add 
                                        # TO DO: ( Upgrade this to handle more difficult case)
                                        if ~np.ma.any( np.abs(bin_rows[search_idx[idx]] - bin_rw_b4) <=3 ) : 
                                            idx_list.append(idx) # idx_list is the index of bin rows/cols
                                            srtd_tmp_cols.append(iter) # Might not be necessary                                               
                                                    
                                        
                            
                            # idx_list is the index of bin rows/cols
                            used_rw_idx = sorted(search_idx[idx_list])
                            
                            # # Check if there is any large discontinuity between consecutive columns                     
                            # stop_row = np.where ( np.diff(bin_rows[used_rw_idx]) > threshold )[0]                        
                            # # Check if there's a jump meaning it should be splitted into different layers
                            # if stop_row.ndim and stop_row.size:
                            #     stop_row = stop_row.item() if len(stop_row) < 2 else stop_row[0].item() # Check this again
                            #     stop_row = min(stop_row,Nx) # Truncate to Nx if greater since search window was deliberately longer
                            # else:
                            #     stop_row = min(Nx,len(used_rw_idx) )                        
                            # used_rw_idx = used_rw_idx[:stop_row+1] # Include stop_row  
                            
                        if debug_plot:
                            plt.figure(1); _ = plt.plot(bin_cols[used_rw_idx], bin_rows[used_rw_idx])     
                           
                        if curr_loc + Nx < loc_idx[-1]:            
                            rows_new.append( np.array(bin_rows[used_rw_idx]) ) # Max loc should be not be included in the current layer  
                            cols_new.append( np.array(bin_cols[used_rw_idx]) )                 
                                
                            # curr_loc += max_loc                         
                        else:                        
                            rows_new.append( np.array(bin_rows[curr_loc:loc_idx[-1]] ) )
                            cols_new.append( np.array(bin_cols[curr_loc:loc_idx[-1]] ) )                   
                        
                        # Set all used bin_rows to NaN
                        bin_rows[used_rw_idx] = np.nan
                        # diff_temp[used_rw_idx] = np.nan
                    
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
        
        
        
        layers_new2 = layers_new.copy()
        layers_new2[np.isnan(layers_new2)] = 0
        
        new_layer_filtered = np.full_like(layers_new, fill_value = np.nan)
        for chan in range(new_layer_filtered.shape[0]):
              new_layer_curr = layers_new2[chan,:]
              if ~np.all(np.isnan(new_layer_curr)) and len(new_layer_curr[~np.isnan(new_layer_curr)]) > 21:
                  new_layer_filtered[chan,:] =  sc_med_filt(new_layer_curr, size=55).astype('int32') #sc_med_filt(z,size=3)
              else:
                  new_layer_filtered[chan,:] = np.nan
                  
                  
        new_layer_filtered [ new_layer_filtered< 0] = np.nan 
        del_idx = np.argwhere(np.sum(new_layer_filtered,axis=1)==0) # Find "all zero" rows              
        new_layer_filtered = np.delete(new_layer_filtered,del_idx,axis = 0) # Delete them
        
        new_layer_filtered [new_layer_filtered==0] = np.nan
        short_layers = np.argwhere( np.sum(np.isnan(new_layer_filtered),axis = 1) > Nx//1.5)
        
        # incomp_wavy_layers = np.argwhere(np.nansum( np.abs(np.diff(new_layer_filtered,axis = 1)),axis=1) > 100 ) & (np.argwhere( np.sum(~np.isnan(new_layer_filtered),axis=1) < round(0.5*Nx))).T #np.nansum( np.abs(np.diff(new_layer_filtered,axis = 1))
        # short_layers = np.append( short_layers, incomp_wavy_layers ) 
        
        new_layer_filtered = np.delete(new_layer_filtered,short_layers,axis = 0) 
        
        model_out =  aa['model_output']
        
        f, axarr = plt.subplots(1,5,figsize=(20,20))
      
        axarr[0].imshow(echo_img.squeeze(),cmap='gray_r')
        axarr[0].set_title( f'Echo {base_name}') #.set_text
        
        axarr[1].imshow(echo_img,cmap='viridis', norm=colors.SymLogNorm(linthresh=0.03, vmin=.85*(echo_img[echo_img>0]).min(), vmax=echo_img.max()) )
        axarr[1].set_title('Orig echo map')
        
        axarr[2].imshow(model_out, cmap='viridis', norm=colors.SymLogNorm(linthresh=0.03, vmin=.85*model_out.min(), vmax=model_out.max()) )
        axarr[2].set_title('Model output') 
        
        axarr[3].imshow(raster, cmap='viridis', norm=colors.SymLogNorm(linthresh=0.03, vmin=.85*raster.min(), vmax=raster.max()) )
        axarr[3].set_title('Binarized output') 
        
        axarr[4].imshow(echo_img.squeeze(),cmap='gray_r' )
        axarr[4].plot(new_layer_filtered.T) 
        axarr[4].set_title('Overlaid_Prediction') 
        
        if 1:
            save_fig_path = os.path.join(base_path,'plotted_images_final',base_name)
            save_fig_path,_ =  os.path.splitext(save_fig_path)
            plt.savefig(save_fig_path+'.png')        
            f.clf()
            plt.close()
        
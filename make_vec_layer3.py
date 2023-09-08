# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 11:52:56 2023
 Script sort_small_list for make_vec_layer
@author: i368o351
"""

import os
import numpy as np
from scipy.io import loadmat
import glob
import matplotlib.pyplot as plt
from make_vec_layerv2 import make_vec_layerv2 
from scipy.ndimage import median_filter as sc_med_filt
from matplotlib import colors

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
base_path = r'Y:/ibikunle/Python_Project/Fall_2021/Predictions_Folder/EchoViT_out/all_files'
echo_path = r'Y:/ibikunle/Python_Project/Fall_2021/all_block_data/Attention_Train_data/Full_size_data/test_data'
for file in glob.glob(os.path.join(base_path,'*.mat')):
    
    aa = loadmat(file) 
    base_name = os.path.basename(file)
    
    if base_name in ['20120330_04_0207_5km.mat']: #'20120330_04_0220_5km.mat',, '20120330_04_0208_5km.mat'
        raster = aa['binary_output']
        echo_img = loadmat(os.path.join(echo_path,base_name))['echo_tmp']
        Nx = echo_img.shape[1]
        
        # vec_layer = make_vec_layerv2(raster,threshold=15)
        threshold=15
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
        
        for curr_short_rw in short_rows:
           
            curr_neigh = rw_uniq[ (np.abs(curr_short_rw - rw_uniq)) <=15 ]  # curr_neigh = rw_uniq[ (np.abs(curr_short_rw - rw_uniq)).argsort()[:5] ]
            
            if np.any(curr_neigh):
                # curr_neigh = curr_neigh [np.abs( np.array(curr_short_rw) - curr_neigh) <= 20 ] # Neighboring rows mst not be farther than 20
                sum_neighbors = sum( [ sum(bin_rows==elem) for elem in curr_neigh])
                
                if sum_neighbors < Nx//5:
                    raster[raster==curr_short_rw] = 0
        
        
        diff_temp = np.argwhere(raster) 
        diff_temp =  np.asarray( sorted(diff_temp, key=lambda x: (x[0],-x[1])) )
        bin_rows,bin_cols = diff_temp[:,0]+1, diff_temp[:,1] # Correct offsets (first and second derivative)    
       
       
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
                    
                    # Notes: "curr_loc" should have been used but that'd return max_col of 0 so curr_loc+1 is used instead
                    
                    max_val = np.min((dy2dx[curr_loc+1:  np.min((curr_loc+Nx+1,loc_idx[-1]-1))  ] ))
                    max_loc = np.where((dy2dx[curr_loc+1:  np.min((curr_loc+Nx+1,loc_idx[-1]-1))  ] ) == max_val  )[0] + 1 # Extra 1 is added to offset the "curr_loc+1"
                    
                    max_loc = max_loc if len(max_loc) < 2 else max_loc[-1] # Find last occurrence
                    max_loc = max_loc.item() # Max loc should be the beginning of the next layer               
                    
                    if np.abs(max_val) < 5: #or max_loc < 5
                        # Let's avoid deleting: Sort and use columns
                        # Check columns
                        tmp_cols = bin_cols[curr_loc:  np.min((curr_loc+Nx+1,loc_idx[-1]-1))  ]
                        srtd_tmp_cols, idx_list =  [], []
                        for idx,elem in enumerate(tmp_cols):
                            if elem not in srtd_tmp_cols:
                                srtd_tmp_cols.append(elem)
                                idx_list.append(idx+1) # Count from 1 instead of 0
                            else:
                                break
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        new_layer_filtered = vec_layer.copy()
        new_layer_filtered[:] = np.nan      
        for chan in range(new_layer_filtered.shape[0]):
              new_layer_curr = vec_layer[chan,:]
              if ~np.all(np.isnan(new_layer_curr)) and len(new_layer_curr[~np.isnan(new_layer_curr)]) > 21:
                  new_layer_filtered[chan,:] =  sc_med_filt(new_layer_curr, size=55).astype('int32') #sc_med_filt(z,size=3)
              else:
                  new_layer_filtered[chan,:] = np.nan
        new_layer_filtered [ new_layer_filtered< 0] = np.nan 
        del_idx = np.argwhere(np.sum(new_layer_filtered,axis=1)==0) # Find "all zero" rows              
        new_layer_filtered = np.delete(new_layer_filtered,del_idx,axis = 0) # Delete them
        
        new_layer_filtered [new_layer_filtered==0] = np.nan
        short_layers = np.argwhere( np.sum(np.isnan(new_layer_filtered),axis = 1) > Nx//1.3)
        
        incomp_wavy_layers = np.argwhere(np.nansum( np.abs(np.diff(new_layer_filtered,axis = 1)),axis=1) > 100 ) & (np.argwhere( np.sum(~np.isnan(new_layer_filtered),axis=1) < round(0.5*Nx))).T #np.nansum( np.abs(np.diff(new_layer_filtered,axis = 1))
        short_layers = np.append( short_layers, incomp_wavy_layers ) 
        
        new_layer_filtered = np.delete(new_layer_filtered,short_layers,axis = 0) 
        
        
        
        f, axarr = plt.subplots(1,4,figsize=(20,20))
      
        axarr[0].imshow(echo_img.squeeze(),cmap='gray_r')
        axarr[0].set_title( f'Echo {base_name}') #.set_text
        
        axarr[1].imshow(echo_img,cmap='viridis', norm=colors.SymLogNorm(linthresh=0.03, vmin=.85*echo_img.min(), vmax=echo_img.max()) )
        axarr[1].set_title('Orig echo map')
        
        axarr[2].imshow(raster, cmap='viridis', norm=colors.SymLogNorm(linthresh=0.03, vmin=.85*raster.min(), vmax=raster.max()) )
        axarr[2].set_title('Model output') 
        
        axarr[3].imshow(echo_img.squeeze(),cmap='gray_r' )
        axarr[3].plot(new_layer_filtered.T) 
        axarr[3].set_title('Overlaid_Prediction') 
        
        if 1:
            save_fig_path = os.path.join(base_path,'plotted_images2',base_name)
            save_fig_path,_ =  os.path.splitext(save_fig_path)
            plt.savefig(save_fig_path+'.png')        
            f.clf()
            plt.close()
        
        
            
        # # def make_vec_layer3(raster):
            
        # Nt,Nx = raster.shape
        # temp1 = raster.T.nonzero()
        # bin_cols,bin_rows = temp1[0],temp1[1]
        
        # # Remove infrequent rows
        # bins, bin_idx, bin_cnt = np.unique(bin_rows, return_index=True, return_counts=True)
        
        # uniq_cols = np.unique(bin_cols)
        
        # layer1 = []
        # for col in uniq_cols:
        #     layer1.append(bin_rows [bin_cols == col])
        
        # col_lengths = [len(item) for item in layer1]
        # max_len, max_idx = np.max(col_lengths), np.argmax(col_lengths)
        
        # vec_layer = np.full( (max_len,Nx), fill_value = np.nan)
        
        # short_col_idx = [ii for ii,xx in enumerate(col_lengths) if xx < max_len ]
        
        # max_col = np.array(layer1[max_idx])
        
        # threshs = np.round( 3* np.exp(0.3*np.linspace(0,20,20)) )
        
        # for col_idx in range(Nx):
        #     curr_short_col = np.array(layer1[col_idx] )
            
        #     if col_idx in short_col_idx:
        #         mult = len(max_col) - len(curr_short_col) 
        #         fix_pos_list = []
        #         # for repeat in range(mult):
        #         ind = abs(max_col - curr_short_col[:,None]) <= 8 #threshs[repeat] 
        #         if np.all(np.diag(ind)):
        #             ind = abs(max_col - curr_short_col[:,None]) <= 5 # Use smallest thresh
                
        #         fix_pos, pos_zero = np.where(~(ind.max(0)))[0], np.where(~(ind.max(0)))[0][0]
        #         other_pos = fix_pos[1:] [np.abs(fix_pos[1:]-fix_pos[0]) > 5]
                
        #         fix_pos_list = np.append(pos_zero, other_pos ) if len(other_pos)>0 else np.array(pos_zero)
                
        #         list_len = fix_pos_list.size
                
        #         fix_pos_list = fix_pos_list if list_len == mult else fix_pos_list[:mult] if list_len > mult else np.append(fix_pos_list, np.zeros((np.abs(list_len-mult),)))
                
        #         final_list = np.array( [item for item in range(max_len) if item not in fix_pos_list ])
                
        #         assert len(final_list) == len(curr_short_col)
        #         vec_layer[final_list,col_idx]  = curr_short_col
                
                
        # vec_layer [vec_layer == 0 ] = np.nan
        # _ = plt.plot(vec_layer.T); _ = plt.gca().invert_yaxis()
        
            
            
            
            
        
        
        
        
        
# -*- coding: utf-8 -*-
"""
Created on Monday Oct 31st 23:58:40 2022

@author: Ibikunle
Weakly supervised dataset
"""
from itertools import groupby
import numpy as np
import os
import glob
from scipy.io import loadmat,savemat
from matplotlib import pyplot as plt
import matplotlib as mpl
from statistics import mode

import mat73

import scipy

from custom_binarize import custom_binarize
from make_vec_layer import make_vec_layer

from scipy.ndimage import median_filter as sc_med_filt
import tensorflow as tf
# import cv2 as cv

######################################################################
# Helper functions
######################################################################

def fix_final_prediction(a, a_final, closeness = 10):
    """
    inputs: a --> raw probabilities
            a_final --> thresholded_probability
            
    output: a_final --> Overwrites input to return binary mask  
    """
    for col_idx in range(a_final.shape[1]):
        
        # Find groups of 0s and 1s
        repeat_tuple = [ (k,sum(1 for _ in groups)) for k,groups in groupby(a_final[:,col_idx]) ]
        # Cumulate the returned index
        rep_locs = np.cumsum([ item[1] for item in repeat_tuple])
        
        # Temporary hack
        rep_locs[-1] = rep_locs[-1] - 1          

        locs_to_fix = [ (elem[1],rep_locs[idx]) for idx,elem in enumerate(repeat_tuple) if elem[0]== 1 and elem[1]>1 ]
        
        for elem0 in locs_to_fix:
            check_idx = list(range(elem0[1]-elem0[0],elem0[1]+1))
            max_loc = check_idx[0] + np.argmax(a[elem0[1]-elem0[0]:elem0[1], col_idx])
            check_idx.remove(max_loc)            
            a_final[check_idx,col_idx] = 0
        
        ## Section to find ones whose index are close and remove those with lower probabilities
        
        # Find groups of 0s and 1s the second time after repeated 1s have been removed.
        repeat_tuple = [ (k,sum(1 for _ in groups)) for k,groups in groupby(a_final[:,col_idx]) ]            
        rep_locs = np.cumsum([ item[1] for item in repeat_tuple]) # Cumulate the returned index
        
        one_locs_idx = [(idx,rep_locs[idx]) for idx,iter in enumerate(repeat_tuple) if iter[0] ==1 ]
        one_locs = [item[1] for item in one_locs_idx] # Just the locs of the 1s 
        
        if np.any( np.diff(one_locs, prepend = 0) < closeness ): # Check if any 1s has index less than 5 to the next 1                 
            
            close_locs_idx = np.where(np.diff(one_locs, prepend = 0) < closeness )[0]                
            
            for item in close_locs_idx:
                # Compare the probs of the "1" before the close 
                check1, check2 = one_locs[item-1]-1, one_locs[item]-1 # Indexing is off by one
                min_chk = check1 if a[check1,col_idx] < a[check2,col_idx] else check2
                a_final[min_chk, col_idx] = 0
        
    
    return a_final   


def create_vec_layer_new(raster,threshold={'constant':25}, debug = False):
    '''              
    Parameters
    ----------
    raster : TYPE: numpy array
        DESCRIPTION: raster vector with lots of zeros (size Nx * Nt)
    threshold : TYPE: int (scalar)
        DESCRIPTION: Determines the minimum jump between 2 consecutive layers

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
    bin_rows,bin_cols = diff_temp[:,0], diff_temp[:,1]    
    bin_rows +=1 # Correct offset
    
    if 'constant' in threshold.keys():
        threshold = threshold['constant'] * np.ones(shape=(10000,))
    else:
        threshold = np.round( list( threshold.values())[0] *np.exp(0.008*np.linspace(1,100,100)) )
           
    #threshold = np.round( threshold*np.exp(0.008*np.linspace(1,100,100)) )
    
    
    # Initialize
    brk_points = [ 0 ] 
    min_jump = 5
  
    vec_layer = np.zeros( shape=(1,Nx) ) # Total number of layers is not known ahead of time
    
    # Initializations
    brk_pt_start = 0;  brk_pt_stop = Nx+5;
    brk_pt_start2,brk_pt_stop2 = None, None
    
    count = 0
    
    while brk_pt_start < len(bin_rows):
        if ( np.diff(bin_rows[brk_pt_start:brk_pt_stop] ) > min_jump ).any():
            curr_max_jump = np.max( np.diff (bin_rows[brk_pt_start:brk_pt_stop]) )            
            curr_max_loc = np.where(np.diff(bin_rows[brk_pt_start:brk_pt_stop]) >= curr_max_jump )[0]
            
            curr_max_loc = np.array([curr_max_loc[-1]]) if len(curr_max_loc)>1 else curr_max_loc
            
            if curr_max_loc  > Nx-2:                
                tmp_res = curr_max_loc
                
            else:                                
                # If less than: there's a break or the current layer is not the entire Nx long.
                # Check if the current jump does not exceed max allowed(i.e threshold). If it does; split into a new layer else attempt to use all Nx-1
                if curr_max_jump > threshold[count]:
                    tmp_res =  curr_max_loc                
                else:
                    # Search further ignoring the current max jump since it's still lower than global threshold
                    next_max = np.max ( np.diff(bin_rows[brk_pt_start+ curr_max_loc.item() :brk_pt_stop]) )
                    tmp_res = curr_max_loc +  np.where(np.diff(bin_rows[brk_pt_start+curr_max_loc.item() :brk_pt_stop]) == next_max) [0]
                
            if len(tmp_res)>1:
                brk_pt_stop = (tmp_res[tmp_res>0][0] + brk_pt_start).item()
            else:
                brk_pt_stop = (tmp_res  + brk_pt_start ).item()
        else:
            if brk_pt_stop < len(bin_rows):                   
               
                tmp_res = np.array([Nx - 1]) 
                brk_pt_stop = (tmp_res  + brk_pt_start ).item()                
            else:
                # Should be the last layer
                brk_pt_stop = len(bin_rows)                               
            
        vec_layer = np.concatenate( (vec_layer,np.zeros(shape=(1,Nx)) ) )
        used_cols = bin_cols[brk_pt_start:brk_pt_stop+1] # Added extra 1 because of zero indexing
        used_rows = bin_rows[brk_pt_start:brk_pt_stop+1 ] # Added extra 1 because of zero indexing
        
        _,used_cols_unq = np.unique(used_cols,return_index=True)
        
        used_cols = used_cols[used_cols_unq]
        used_rows = used_rows[used_cols_unq]
         
        vec_layer[-1, used_cols ] = used_rows                                   
        brk_points.append( (brk_pt_start,brk_pt_stop,brk_pt_stop-brk_pt_start, list(used_rows) ) )
        
        if brk_pt_start2 and brk_pt_stop2:
            vec_layer = np.concatenate( (vec_layer,np.zeros(shape=(1,Nx)) ) )
            used_cols = bin_cols[brk_pt_start2:brk_pt_stop2 +1 ]
            used_rows = bin_rows[brk_pt_start2:brk_pt_stop2 +1 ]                
            _,used_cols_unq = np.unique(used_cols,return_index=True)
            
            used_cols = used_cols[used_cols_unq]
            used_rows = used_rows[used_cols_unq]
             
            vec_layer[-1, used_cols ] = used_rows                                   
            brk_points.append( (brk_pt_start2,brk_pt_stop2,brk_pt_stop2-brk_pt_start2, list(used_rows) ) )
            
            brk_pt_start, brk_pt_stop = brk_pt_start2, brk_pt_stop2 # Set the new brk_points to the latest one                                               
        
        if debug:
            # To be removed after debugging
            brk_pt_stop = brk_pt_stop-1 if brk_pt_stop == len(bin_rows) else brk_pt_stop         
            print('=================================================================================================')
            print(f'Current layer have {brk_pt_stop - brk_pt_start} elements')
            print(f'Layer {count}: Ends with { bin_rows[brk_pt_stop]} ({brk_pt_stop}), next layer starts with {bin_rows[brk_pt_stop+1]} {(brk_pt_stop+1)}')
            print(f'layer jump is {bin_rows[brk_pt_stop+1]} - {bin_rows[brk_pt_stop] } = {bin_rows[brk_pt_stop+1] - bin_rows[brk_pt_stop] } ')
            print('=================================================================================================')
            
        
        brk_pt_start = brk_pt_stop + 1
        brk_pt_stop = brk_pt_start + Nx + 5 # Adding extra one to complete 64 and realizing Python indexing w/o last element
        
        brk_pt_stop2 = brk_pt_start2 = None           
         
        count +=1

    return vec_layer
    
    
    
def create_vec_layer(raster,threshold={'constant':5}):
    '''              
    Parameters
    ----------
    raster : TYPE: numpy array
        DESCRIPTION: raster vector with lots of zeros (size Nx * Nt)
    threshold : TYPE: int (scalar)
        DESCRIPTION: Determines the minimum jump between 2 consecutive layers

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
    bin_rows,bin_cols = diff_temp[:,0], diff_temp[:,1]    
    bin_rows +=1 # Correct offset
    
    if 'constant' in threshold.keys():
        threshold = threshold['constant'] * np.ones(shape=(100,))
    else:
        threshold = np.round( list( threshold.values())[0] *np.exp(0.008*np.linspace(1,100,100)) )
           
    #threshold = np.round( threshold*np.exp(0.008*np.linspace(1,100,100)) )
    #threshold = 7

    #brk_points = [ idx for (idx,value) in enumerate(np.diff(bin_rows)) if value > threshold ]   #np.diff(bin_rows)
    #brk_pt_chk = [ (idx,bin_rows[idx],value) for (idx,value) in enumerate(np.diff(bin_rows)) if value > threshold] 
    
    
    # Initialize
    brk_points = [ 0 ] 
    vec_layer = np.zeros( shape=(1,Nx) ) # Total number of layers is not known ahead of time
    
    # Initializations
    brk_pt_start = 0;  brk_pt_stop = Nx;
    brk_pt_start2,brk_pt_stop2 = None, None
    
    count = 0
    
    while brk_pt_start < len(bin_rows) :
        if ( np.diff(bin_rows[brk_pt_start:brk_pt_stop] ) > threshold[count] ).any():
            tmp_res = np.where(np.diff(bin_rows[brk_pt_start:brk_pt_stop]) > threshold[count] )[0] #int(threshold[count])
            if len(tmp_res)>1:
                brk_pt_stop = (tmp_res[tmp_res>0][0] + brk_pt_start).item()
            else:
                brk_pt_stop = (tmp_res  + brk_pt_start ).item()
        else:
            if brk_pt_stop < len(bin_rows):                    
                brk_pt_stop2 = brk_pt_stop + Nx
                tmp_res = np.where(np.diff(bin_rows[brk_pt_start:brk_pt_stop2]) > threshold[count] )[0] 
                
                if len(tmp_res) == 0:
                    max_diff = np.max( np.diff(bin_rows[brk_pt_start:brk_pt_stop2]) )
                    tmp_res = np.where(np.diff(bin_rows[brk_pt_start:brk_pt_stop2]) == max_diff )[0] 
                    tmp_res = tmp_res[::-1]

                brk_pt_stop2 =  (tmp_res[tmp_res>0][0] + brk_pt_start).item() if len(tmp_res)>1 else  (tmp_res  + brk_pt_start ).item()
                
            else:
                # Should be the last layer
                brk_pt_stop2 = len(bin_rows)                                 
            
            brk_pt_stop = brk_pt_stop2 - Nx # This might not be correct
            brk_pt_start2 = brk_pt_stop + 1 if brk_pt_stop2 < len(bin_rows) else brk_pt_start
                
           
        vec_layer = np.concatenate( (vec_layer,np.zeros(shape=(1,Nx)) ) )
        used_cols = bin_cols[brk_pt_start:brk_pt_stop+1] # Added extra 1 because of zero indexing
        used_rows = bin_rows[brk_pt_start:brk_pt_stop+1 ] # Added extra 1 because of zero indexing
        
        _,used_cols_unq = np.unique(used_cols,return_index=True)
        
        used_cols = used_cols[used_cols_unq]
        used_rows = used_rows[used_cols_unq]
         
        vec_layer[-1, used_cols ] = used_rows                                   
        brk_points.append( (brk_pt_start,brk_pt_stop,brk_pt_stop-brk_pt_start, list(used_rows) ) )
        
        if brk_pt_start2 and brk_pt_stop2:
            vec_layer = np.concatenate( (vec_layer,np.zeros(shape=(1,Nx)) ) )
            used_cols = bin_cols[brk_pt_start2:brk_pt_stop2 +1 ]
            used_rows = bin_rows[brk_pt_start2:brk_pt_stop2 +1 ]                
            _,used_cols_unq = np.unique(used_cols,return_index=True)
            
            used_cols = used_cols[used_cols_unq]
            used_rows = used_rows[used_cols_unq]
             
            vec_layer[-1, used_cols ] = used_rows                                   
            brk_points.append( (brk_pt_start2,brk_pt_stop2,brk_pt_stop2-brk_pt_start2, list(used_rows) ) )
            
            brk_pt_start, brk_pt_stop = brk_pt_start2, brk_pt_stop2 # Set the new brk_points to the latest one                                               
        
        brk_pt_start = brk_pt_stop + 1
        brk_pt_stop = brk_pt_start + Nx + 5 # Adding extra one to complete 64 and realizing Python indexing w/o last element
        
        brk_pt_stop2 = brk_pt_start2 = None           
         
        count +=1
    return vec_layer

######################################################################
# Paths
######################################################################
base_path = r'X:\ct_data\snow\2012_Greenland_P3'  #Y:\ibikunle\Python_Project\Fall_2021\Model_and_weights\wavelet_ouput\drive-download-20221006T191142Z-001' #d Single
out_dir = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Weak_labels_data2'  #out_dir = r'X:\public\data\temp\internal_layers\NASA_OIB_test_files\image_files\snow\OLD_2012_Greenland_P3\predictions_folder'

# Set the default color cycle
ab = list( mpl.cycler(mpl.rcParams['axes.prop_cycle']) )
clr = [ item['color'] for item in ab if item['color'] != '#7f7f7f' ] # Remove illegible color
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=clr) 


if not os.path.exists(out_dir):
    os.mkdir(out_dir) 

model_save_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data\Res50_pretrained_model\Res50_pretrained_model_98.51_02_November_22_1845.h5'
# model_save_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data\EchoViT_paper\EchoViT_paper_OKAY_binary_30_December_22_2204.h5'


base_echo_path = os.listdir(base_path)
base_echo_path = sorted( [ elem for elem in base_echo_path if 'dataset' in elem ] )


######################################################################
# Flags
######################################################################
plot_pred = True    
save_pred = True #True
save_plot = True # Plot pred must be True for this

rmv_island = False

######################################################################
# Load model
######################################################################
model = tf.keras.models.load_model(model_save_path, compile=False)



######################################################################
# Perform thresholding and layer extraction
######################################################################

max_Nt = 1664

for segment_folder in base_echo_path:
    
    for folder in os.listdir( os.path.join(base_path,segment_folder) ) :
        print (f" Now in folder {folder}")
    
        
        for idx,file in enumerate(glob.glob(os.path.join(base_path,segment_folder,folder,'*.mat') ) ):  
            id = segment_folder.find('dataset_')
            bn1,bn2 = os.path.splitext( os.path.basename(file) )
            base_name = bn1 + '_weak' + segment_folder[id:] + bn2
            
            if 'lat' not in base_name and not os.path.isfile(os.path.join(out_dir,folder,base_name)): # Check if file already exist or it's a lon/lat file # not os.path.isfile( os.path.join(out_dir,folder,base_name)) and 
            
                curr_img = mat73.loadmat(file)
                curr_data = curr_img['Data']
                
                curr_Nt,curr_Nx = curr_data.shape
                
                if curr_Nt < max_Nt:
                    gap = max_Nt - curr_Nt
                    curr_data = np.append( curr_data, np.zeros((gap,curr_Nx)), axis=0 )
                else:
                    curr_data = curr_data[:max_Nt,:] 
                
                # curr_Nt = curr_Nt if curr_Nt % 32 == 0 else ((curr_Nt//32) )*32                 
                # curr_Nx = curr_Nx if curr_Nx % 32 == 0 else ((curr_Nx//32))*32 -1
                
                filter_x,filter_y = int(curr_Nt//60), int(curr_Nx//5)
                conv_filter = np.ones(shape=(filter_x,filter_y ))
                
                curr_prob_map = model.predict(np.expand_dims(np.expand_dims(curr_data,axis = -1),axis =0 ) )
                curr_prob_map  = curr_prob_map.squeeze()
                
                curr_prob_map[max_Nt-1:,:] = 0 # Remove prediction artifact
                
                # curr_prob_map = 0.6*curr_prob_map + .4* ( curr_prob_map * curr_data)
                
                # threshold_check_value = sum((np.sum(~np.isnan(curr_img['layers_vector']), axis = 1)))//curr_Nx    
                # binarize_threshold = np.percentile(curr_prob_map,77) if threshold_check_value >= curr_Nx*3 else np.percentile(curr_prob_map,95)
                # binarize_threshold =  mod_val + 0.17*(np.max(curr_prob_map) - mod_val) if threshold_check_value >= 20 else np.percentile(curr_prob_map,98)
                # threshold_check_value = 100* ( np.sum(curr_prob_map> 1.5*np.std(curr_prob_map))/np.prod(curr_prob_map.shape) )
                
                C2 = curr_prob_map.copy()
                C2 [np.isnan(C2)] = 0
                _,bins = np.histogram( C2.ravel() )              
                binarize_threshold =  0.8*bins[2] + 0.2*bins[3] 
                
                if binarize_threshold > 0.01:
                
                    C0 = np.where(curr_prob_map>binarize_threshold,1,0)
                    C02 = np.copy(C0)
                    res0_island_rmv = C02.copy() ; 
                    
                    if rmv_island:                        # Remove islands and discontinuities in thresholded predictions
                        
                        # conv_vals = cv.filter2D(res0_island_rmv, -1, conv_filter, borderType=cv.BORDER_CONSTANT)  
                        conv_vals = scipy.signal.convolve2d(res0_island_rmv,conv_filter,mode='same',boundary='symm') 
                        res0_island_rmv[conv_vals < np.max(conv_vals)//4] = 0 # Remove island predictions
                    
                       
                    curr_prob_thresh = custom_binarize(curr_prob_map,res0_island_rmv,closeness = 15)
                    
                    dim0 = curr_prob_thresh.shape[0]
                    
                    vec_layer = make_vec_layer( np.arange(1,dim0+1).reshape(dim0,1) * curr_prob_thresh, threshold={'constant':40})
                    new_layer_filtered = vec_layer.copy() 
                    
                    vec_layer [ vec_layer <= 0] = np.nan                  
                    
                    
                    new_layer_filtered[:] = np.nan      
                    for chan in range(new_layer_filtered.shape[0]):
                        new_layer_curr = vec_layer[chan,:]
                        if ~np.all(np.isnan(new_layer_curr)) and len(new_layer_curr[~np.isnan(new_layer_curr)]) > 21:
                            new_layer_filtered[chan,:] =  sc_med_filt(new_layer_curr, size=35).astype('int32') #sc_med_filt(z,size=3)
                        else:
                            new_layer_filtered[chan,:] = np.nan
                              
                              
                    new_layer_filtered [ new_layer_filtered< 0] = np.nan 
                    
                    del_idx = np.argwhere(np.sum(new_layer_filtered,axis=1)==0) # Find "all zero" rows              
                    new_layer_filtered = np.delete(new_layer_filtered,del_idx,axis = 0) # Delete them
                    
                    new_layer_filtered [new_layer_filtered==0] = np.nan
                    short_layers = np.argwhere( np.sum(np.isnan(new_layer_filtered),axis = 1) > round(.75*curr_Nx) )
                    incomp_wavy_layers = np.argwhere(np.nansum( np.abs(np.diff(new_layer_filtered,axis = 1)),axis=1) > 100 ) & (np.argwhere( np.sum(~np.isnan(new_layer_filtered),axis=1) < round(0.5*curr_Nx))).T #np.nansum( np.abs(np.diff(new_layer_filtered,axis = 1))

                    short_layers = np.append( short_layers, incomp_wavy_layers ) 
                    
                    
                    new_layer_filtered = np.delete(new_layer_filtered,short_layers,axis = 0)
                    
                    new_layer_filtered [ new_layer_filtered > max_Nt ] = 0
                    
                    if plot_pred: 
                        
                        fig,axarr = plt.subplots(1,6,figsize=(20,20))           
                        
                        axarr[0].imshow(curr_data,cmap='gray_r')
                        axarr[0].set_title(f'{os.path.splitext(base_name)[0]}')
                        
                        axarr[1].imshow(curr_data,cmap='viridis')
                        axarr[1].set_title('Orig echo map')
                        
                        axarr[2].imshow(curr_prob_map)
                        axarr[2].set_title('Model direct output')
                        
                        axarr[3].imshow(curr_prob_thresh)
                        axarr[3].set_title('Binarized and thresholded')               
                        
                        axarr[4].imshow(curr_data,cmap='gray_r')
                        axarr[4].plot(vec_layer.T)
                        axarr[4].set_title('Overlayed prediction')
                        
                        axarr[5].imshow(curr_data,cmap='gray_r')
                        axarr[5].plot(new_layer_filtered.T)
                        axarr[5].set_title('Filtered Overlayed prediction')
                        
                        
                    if save_pred:
                        if not os.path.exists(os.path.join(out_dir,folder)):
                            os.mkdir(os.path.join(out_dir,folder)) 
                                              
                        save_path = os.path.join(out_dir,folder,base_name)
                        
                        if save_plot:
                            if not os.path.exists(os.path.join(out_dir,folder,'plotted_images')):
                                os.mkdir(os.path.join(out_dir,folder,'plotted_images')) 
                            
                            save_fig_path = os.path.join(out_dir,folder,'plotted_images', base_name)
                            save_fig_path,_ =  os.path.splitext(save_fig_path)
                            plt.savefig(save_fig_path+'.png')
                            
                        fig.clf()
                        plt.close()
                        
                        # Create new raster and semantic_seg
                        raster = np.zeros_like(curr_data) 
                        semantic_seg2 = np.zeros_like(curr_data) 
                        for row_idx in range(new_layer_filtered.shape[0]):
                            col_idx = np.argwhere(~np.isnan(new_layer_filtered[row_idx]))
                            raster_vals =  new_layer_filtered[row_idx][col_idx].astype('int')
                            
                            raster_vals -=1   
                            
                            raster[ raster_vals, col_idx ] = 1
                            
                            for iter,raster_row in enumerate(raster_vals):
                                semantic_seg2[ raster_row.item(): , col_idx[iter].item() ] = row_idx+1
                        
    
                        out_dict= {}
                        out_dict['echo_tmp'] = curr_data
                        out_dict['vec_layer'] = new_layer_filtered
                        out_dict['raster'] = curr_prob_thresh
                        out_dict['raster2'] = raster
                        out_dict['semantic_seg2'] = semantic_seg2
                        
                        out_dict['layers_vector'] = curr_img['layers_vector']
                        out_dict['original_frame'] = curr_img['original_frame']
                        out_dict['Elevation'] = curr_img['Elevation']                        
                                                
                        out_dict['Time'] = curr_img['Time']
                        
                        try:
                            out_dict['weather_data'] = curr_img['weather_data']
                        except:
                            pass                      
                        
                        print(f" Now saving output for {base_name} in {save_path}")
                        print()
                        print(f" {(idx)+1} of { len(glob.glob(os.path.join(base_path,segment_folder,folder,'*.mat')))  } ")
                        savemat(save_path,out_dict)
                        
    
                       
               

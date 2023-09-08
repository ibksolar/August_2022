# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 11:50:40 2022

@author: Ibikunle
Script to binarize Wavelet model output and create vec_layer
"""
from itertools import groupby
import numpy as np
import os
import glob
from scipy.io import loadmat,savemat
from matplotlib import pyplot as plt
import re


######################################################################
# Paths
######################################################################
base_path = r'Y:\ibikunle\Python_Project\Fall_2021\Model_and_weights\wavelet_ouput\drive-download-20221006T191142Z-001' #d Single
# out_dir = r'Y:\ibikunle\Python_Project\Fall_2021\Model_and_weights\wavelet_ouput\predictions_folder'
out_dir = r'X:\public\data\temp\internal_layers\NASA_OIB_test_files\image_files\snow\OLD_2012_Greenland_P3\predictions_folder'

base_echo_path = r'X:\public\data\temp\internal_layers\NASA_OIB_test_files\image_files\snow\OLD_2012_Greenland_P3\2012_Greenland_P3\frames_001_243_20120330_04\image'

######################################################################
# Flags
######################################################################
plot_pred = False    
save_pred = True 


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
    
    ### Create Old vec_layer function
    def create_vec_layer_old(raster):
        import itertools
        vec_layer = []
        for iter in range(raster.shape[1]):
            temp = np.nonzero(raster[:,iter])
            vec_layer.append(temp[0])
            
        return np.array(list(itertools.zip_longest(*vec_layer, fillvalue=0)))
    ##############################################################################    
    
    
    
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


    #brk_points = [-1] + brk_points + [len(bin_rows)]        
    # brk_points[0] = -1
    # num_layers = len(brk_points) - 1        
    # vec_layer = np.zeros( ( num_layers, raster.shape[-1] ) ) 
    
    # for iter in range(num_layers):
    #     start_idx , stop_idx = brk_points[iter]+1 , brk_points[iter+1] + 1            
    #     vec_layer[iter, bin_cols[start_idx:stop_idx] ] = bin_rows[start_idx:stop_idx ]   
        
        
        
    return vec_layer


######################################################################
# Perform thresholding and layer extraction
######################################################################


base_dirs = os.listdir(base_path)

for folder in base_dirs:
    print (f" Now in folder {folder}")

    
    for idx,file in enumerate(glob.glob(os.path.join(base_path,folder,'*.mat') ) ):
        
        base_name = os.path.basename(file)
        
        if not os.path.isfile( os.path.join(out_dir,folder,base_name)) : # Check if file already exist
        
            curr_img = loadmat(file)
            curr_prob = curr_img['img']
            C0 = np.where(curr_prob>0.4,1,0)
            C02 = np.copy(C0)
            
            curr_prob_thresh = fix_final_prediction(curr_prob,C02,closeness = 20)
            
            dim0 = curr_prob_thresh.shape[0]
            
            vec_layer = create_vec_layer( np.arange(1,dim0+1).reshape(dim0,1) * curr_prob_thresh, threshold={'constant':5})
            
            del_idx = np.argwhere(np.sum(vec_layer,axis=1)==0) # Find "all zero" rows
            vec_layer = np.delete(vec_layer,del_idx,axis = 0) # Delete them
            
            vec_layer [vec_layer==0] = np.nan
            
            
            
            echo_idx = re.search(r'\d+',base_name).group()        
            curr_echo = os.path.join(base_echo_path,f"image_{echo_idx}.mat")        
            
            try:
                curr_echo_default = loadmat(curr_echo)
                curr_echo = curr_echo_default['data']
                curr_layer = curr_echo_default['layer'] 
            except:
                continue        
            
            if plot_pred: 
                
                f,axarr = plt.subplots(1,5,figsize=(20,20))           
                
                axarr[0].imshow(curr_echo,cmap='gray_r')
                axarr[0].set_title('Before threshold')
                
                axarr[1].imshow(C0)
                axarr[1].set_title('Before threshold')
                
                # axarr[1].imshow(curr_prob_thresh,cmap='gray_r')
                # axarr[1].set_title('Prediction')
                
                axarr[2].plot(vec_layer.T)
                axarr[2].set_title('Predicted layer')
                axarr[2].invert_yaxis()              
                
                axarr[3].plot(curr_layer.T)
                axarr[3].set_title('GT layer')
                axarr[3].invert_yaxis()  
                
                axarr[4].imshow(curr_echo,cmap='gray_r')
                axarr[4].plot(vec_layer.T)
                axarr[4].set_title('Overlayed prediction')
                
                
            if save_pred:
                if not os.path.exists(os.path.join(out_dir,folder)):
                    os.mkdir(os.path.join(out_dir,folder)) 
                                      
                save_path = os.path.join(out_dir,folder,base_name)
                
                out_dict= {}
                out_dict['echo_img'] = curr_echo
                out_dict['GT_layer'] = curr_layer
                out_dict['Predicted_layer'] = vec_layer
                out_dict['Prob_map'] = curr_prob_thresh
                
                print(f" Now saving output for {base_name} in {save_path}")
                print()
                print(f" {(idx)+1} of { len(glob.glob(os.path.join(base_path,folder,'*.mat')))  } ")
                savemat(save_path,out_dict)
               
           

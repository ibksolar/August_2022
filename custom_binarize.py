# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 15:24:43 2022

@author: i368o351
"""
from itertools import groupby
import numpy as np
from scipy.ndimage import median_filter as sc_med_filt


######################################################################
# Helper function
######################################################################

def custom_binarize(prob_map, thresh_prob_map, closeness = 10, return_segment = False):
    """
    inputs: prob_map --> raw probabilities
            thresh_prob_map --> thresholded_probability
            
    output: thresh_prob_map --> Overwrites input to return binary mask  
    """
    min_layer_loc = 20
    final_return_segment = np.zeros_like(prob_map)
    
    thresh_prob_map = sc_med_filt(thresh_prob_map, size=(1,25))
    
    Nt,Nx = prob_map.shape
            
    for col_idx in range(thresh_prob_map.shape[1]):
        
        # Find groups of 0s and 1s
        repeat_tuple = [ (k,sum(1 for _ in groups)) for k,groups in groupby(thresh_prob_map[:,col_idx]) ]
        # Cumulate the returned index
        rep_locs = np.cumsum([ item[1] for item in repeat_tuple])
        
        if len(rep_locs)>1: # bad columns have less or just one 
        # Temporary hack
            rep_locs[-1] = rep_locs[-1] - 1   
            
            short_one_locs = [(rep_locs[count],iter[1]) for count,iter in enumerate(repeat_tuple) if iter[0]==1 and iter[1]<=closeness]            
            for each in short_one_locs:
                thresh_prob_map [ each[0]-each[1]:each[0], col_idx] = 1            
            
                        
            repeat_tuple = [ (k,sum(1 for _ in groups)) for k,groups in groupby(thresh_prob_map[:,col_idx]) ]
            rep_locs = np.cumsum([ item[1] for item in repeat_tuple])
            locs_to_fix = [ (elem[1],rep_locs[idx]) for idx,elem in enumerate(repeat_tuple) if elem[0]== 1 and elem[1]>1 ]
            
            for elem0 in locs_to_fix:
                check_idx = list(range(elem0[1]-elem0[0],  elem0[1]+1 ) )
                max_loc = check_idx[0] + np.argmax(prob_map[elem0[1]-elem0[0]:elem0[1], col_idx])
                check_idx.remove(max_loc)
                if Nt in check_idx:
                    check_idx.remove(Nt)
                    
                thresh_prob_map[check_idx,col_idx] = 0
            
            ## Section to find ones whose index are close and remove those with lower probabilities
            
            # Find groups of 0s and 1s the second time after repeated 1s have been removed.
            repeat_tuple = [ (k,sum(1 for _ in groups)) for k,groups in groupby(thresh_prob_map[:,col_idx]) ]            
            rep_locs = np.cumsum([ item[1] for item in repeat_tuple]) # Cumulate the returned index
            
            one_locs_idx = [(idx,rep_locs[idx]) for idx,iter in enumerate(repeat_tuple) if iter[0] ==1 ]
            one_locs = [item[1] for item in one_locs_idx] # Just the locs of the 1s 
            
            if np.any( np.diff(one_locs, prepend = 0) < closeness ): # Check if any 1s has index less than 5 to the next 1                 
                
                close_locs_idx = np.where(np.diff(one_locs, prepend = 0) < closeness )[0]                
                
                for item in close_locs_idx:
                    # Compare the probs of the "1" before the close 
                    check1, check2 = one_locs[item-1]-1, one_locs[item]-1 # Indexing is off by one
                    min_chk = check1 if prob_map[check1,col_idx] < prob_map[check2,col_idx] else check2
                    thresh_prob_map[min_chk, col_idx] = 0
            
            # If return_segmentation_result
            if return_segment:
                
                repeat_tuple = [ (k,sum(1 for _ in groups)) for k,groups in groupby(thresh_prob_map[:,col_idx]) ]
                # Cumulate the returned index
                rep_locs = np.cumsum([ item[1] for item in repeat_tuple])
                one_locs_idx = [ rep_locs[idx] for idx,iter in enumerate(repeat_tuple) if iter[0] ==1 and rep_locs[idx] > min_layer_loc]
                for iter in range(len(one_locs_idx)):
                    final_return_segment[ one_locs_idx[iter]: min( one_locs_idx[-1]+20,Nt-1), col_idx ] = iter+1
            
            
    return thresh_prob_map if not return_segment else (thresh_prob_map,final_return_segment)
    
    ##############################################################################    
    
# Example usage

# curr_img = loadmat(file)  --> MAT file from model prediction (Debvrat's Google Drive)
# curr_prob = curr_img['img']
# C0 = np.where(curr_prob>0.4,1,0)
# C02 = np.copy(C0)

# curr_prob_thresh = custom_binarize(curr_prob,C02,closeness = 20)
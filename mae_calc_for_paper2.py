# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 07:58:44 2022

@author: i368o351
"""

"""
compares each GT layer with each predicted layer, and finds the least mae for that GT layer.
additionally, drop out any predicted layer that has been compared with GT
exports mae of each GT layer in each file
also exports the overall mae for an experiment across the entire test set
"""

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import glob


base_path = r'Y:\Python_Project\Fall_2021\Predictions_Folder\DL_models_predictions_folder_final'
levels = ['L1','L2','L3']

res = {}
res['L1'] = {}
res['L2'] = {}
res['L3'] = {}

for level in levels:    
    for model in os.listdir( os.path.join(base_path,level) ):
        all_files =  glob.glob( os.path.join(base_path,level,model,'*.mat') )
        
        mae_per_blk = np.zeros( (30,len(all_files)))   # Dimension-> Num_layers x Num_blocks
        
        file_idx = -1
        
        for file in all_files:
            file_idx += 1
            temp = loadmat(file)
            
            GT_layer = temp['GT_layer']
            pred_layer1 = temp['filtered_vec_layer']
            
            _,Nx = pred_layer1.shape

            all_nan_layer = np.argwhere( np.all(np.isnan(GT_layer),axis = 1) )
            GT_layer = np.delete(GT_layer, all_nan_layer, axis = 0)  
            
            short_layers = np.argwhere( np.sum(np.isnan(pred_layer1),axis = 1) > round(.7*Nx) )
            pred_layer = np.delete(pred_layer1,short_layers,axis = 0)            
            
            
            ## Metrics and check
            top_5_mae = np.full( (30,5), fill_value=np.nan )
            top_5_idx = np.full( (30,5), fill_value=np.nan )
            
            tent_MAE_idx = np.full( (GT_layer.shape[0]), fill_value=np.nan )
            tent_MAE_val = np.full( (GT_layer.shape[0]), fill_value=np.nan )
            
            # Compare each GT layer
            for GT_lay_idx in range(GT_layer.shape[0]):
                
                curr_GT_layer = np.ma.array(GT_layer[GT_lay_idx,:], mask = np.isnan(GT_layer[GT_lay_idx,:]) )
                curr_pred_layer = np.ma.array(pred_layer, mask = np.isnan(pred_layer) )
                
                errs = curr_GT_layer - curr_pred_layer
                errs =  np.mean( np.abs( errs), axis=1 )
                
                top_n = np.min((GT_layer.shape[0], pred_layer.shape[0], 5)) # What is the minimum possible?
                
                top_5_idx[GT_lay_idx,:top_n] = np.argsort(errs)[:top_n]
                top_5_mae[GT_lay_idx,:top_n] = errs[ np.argsort(errs)[:top_n] ]
                
                tent_MAE_idx[GT_lay_idx]  = np.argmin(errs)
                tent_MAE_val[GT_lay_idx] = np.nanmin(errs)                
                
            repeating_idx = np.unique( [ int(elem) for elem in tent_MAE_idx if list(tent_MAE_idx).count(elem)>1 ] )
            
            if np.any(repeating_idx):
                
                for iter in range(len(repeating_idx)):
                    rep_idx = repeating_idx[iter]
                    del_val = np.max( tent_MAE_val[np.where(tent_MAE_idx == rep_idx)] )
                
                    del_idx =  np.argwhere(tent_MAE_val == del_val) 
                    del_idx = del_idx[0] if len(del_idx)>1 else del_idx
                    
                    tent_MAE_idx = np.delete(tent_MAE_idx,del_idx.item(), axis = 0)
                    tent_MAE_val = np.delete(tent_MAE_val,del_idx.item(), axis = 0)
            
            if np.all( sorted(tent_MAE_idx)== tent_MAE_idx ):
                print(f'Saving: Level{level}->Model{model}  -->{os.path.basename(file)} ')
                mae_per_blk[0:len(tent_MAE_val),file_idx] = tent_MAE_val
            else:
                print(f'File {os.path.basename(file)} not included')
                wrong_loc = np.argwhere( np.diff(tent_MAE_idx) > 1)
                print(f'ERROR: Level{level}_Model_{model}_{os.path.basename(file)} has wrong order in {wrong_loc}')
                
        res[level][model] = mae_per_blk
            
            

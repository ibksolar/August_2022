# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 10:39:45 2022

@author: i368o351
"""


from tensorflow.keras import layers
from tensorflow import keras
from keras import backend as K
import math
import cv2 as cv

import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

import os
import random
from scipy.io import loadmat,savemat
from scipy.ndimage import median_filter as sc_med_filt
from scipy.signal import butter,filtfilt, medfilt

from keras.metrics import MeanIoU
from sklearn.metrics import roc_auc_score

import glob
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,ReduceLROnPlateau
import segmentation_models as sm
from datetime import datetime
# from focal_loss import SparseCategoricalFocalLoss



##==============================================================================##
## GPU Config
##==============================================================================##

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      #tf_config = tf.ConfigProto(allow_soft_placement=False)
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)  
tf.keras.mixed_precision.set_global_policy('mixed_float16')
##############################################################################


##==============================================================================##
##  Use WandB: Yes or no?
##==============================================================================##
time_stamp = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
use_wandb = False
if use_wandb:
    ## WandB config
    import wandb
    from wandb.keras import WandbCallback
    wandb.init( project="my-test-project", entity="ibksolar", name='Seg_Predict'+time_stamp,config ={})
    config = wandb.config
else:
    config={}

try:
    fname = ipynbname.name()
except:
    fname = os.path.splitext( os.path.basename(__file__) )[0]
finally:
    print ('Could not automatically find file path')
    fname = 'blank'
##############################################################################
## Important Flags
decimated_model = False

run_predictions = True 
save_predictions = False # run_predictions needs to be "True" for save_predictions

# Typically, only one of the 2 below can be "True" at a time
binary_viz = True # Vizualize predictions for binary model
segmentation_viz = False # Vizualize predictions for segmentation model


##==============================================================================##
## Useful custom loss and metrics
##==============================================================================##

def dice_coef(y_true, y_pred):
    y_true = K.squeeze(y_true, axis = -1)
    y_pred = K.squeeze(y_pred, axis = -1)        
        
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum( y_true_f * y_pred_f)
    return ( 2.0 * intersection +1.0)/ (K.sum(y_true_f)  + K.sum(y_pred_f) + 1.0 )

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum( y_true_f * y_pred_f)
    return (  intersection +1.0)/ (K.sum(y_true_f)  + K.sum(y_pred_f) - intersection + 1.0 )

def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
##############################################################################

config['num_patches'] = 416
config['embed_dim'] = 64

class PatchEncoder(layers.Layer):        
    def __init__(self, num_patches = config['num_patches'], embed_dim = config['embed_dim'], **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=embed_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=embed_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    # Override function to avoid error while saving model
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {               
                "num_patches": config['num_patches'],
                "embed_dim": config['embed_dim'],
            }
        )
        return config


##==============================================================================##
## Final prediction correcting functions
##==============================================================================##

from itertools import groupby

def fix_final_prediction(a, a_final, closeness = 15):
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
        vec_layer = create_vec_layer(res0_hard_threshold, 10)

    '''
    
    #TO DO: Check type of raster; should be numpy array
    
    diff_temp = np.argwhere(raster) #raster.nonzero()
    Nx = raster.shape[-1]
    bin_rows,bin_cols = diff_temp[:,0], diff_temp[:,1]    
    bin_rows +=1 # Correct offset
    
    min_jump = 5
    
    if 'constant' in threshold.keys():
        threshold = threshold['constant'] * np.ones(shape=(1000,))
    else: # exponentially increasing threshold. 
          #  TO DO: Add two forms of exponential: fast increasing and slowly increasing
        threshold = np.round( list( threshold.values())[0] *np.exp(0.008*np.linspace(1,1000,1000)) )
        
    # Initialize
    brk_points = [ 0 ] 
    vec_layer = np.zeros( shape=(1,Nx) ) # Total number of layers is not known ahead of time
    
    # Initializations
    brk_pt_start = 0; 
    brk_pt_stop = Nx + 3; # Adding extra 3 to Nx to force it to detect a "jump" in rows (i.e include some of the next layer )
    brk_pt_start2,brk_pt_stop2 = None, None
    
    count = 0
    
    while brk_pt_start < len(bin_rows) :
        if ( np.diff(bin_rows[brk_pt_start:brk_pt_stop] ) >= min_jump ).any():
            
            max_jump = np.max( np.diff(bin_rows[brk_pt_start:brk_pt_stop]) )
            max_jump = max(max_jump, min_jump)
            
            threshold_to_use = threshold[count] if (np.diff(bin_rows[brk_pt_start:brk_pt_stop] ) >= threshold[count]).any() else max_jump #and threshold[count]>max_jump 
            
            exact_brk_point = np.where(np.diff(bin_rows[brk_pt_start:brk_pt_stop]) >= threshold_to_use )[0] #int(threshold[count])
            if len(exact_brk_point)>1:
                brk_pt_stop = (exact_brk_point[exact_brk_point>0][-1] + brk_pt_start).item()
            else:
                brk_pt_stop = (exact_brk_point  + brk_pt_start ).item()
        else:
            if brk_pt_stop < len(bin_rows):     
                
                # There was no "jump" in the last Nx or difference was slightly lesser than threshold
                # Add extra Nx to brk_pt_stop to create brk_pt_stop2
                
                brk_pt_stop2 = brk_pt_stop + Nx  # New initial value for brk_pt_stop2
                exact_brk_point = np.where(np.diff(bin_rows[brk_pt_start:brk_pt_stop2]) > threshold[count] )[0] 
                
                max_diff = np.max( np.diff(bin_rows[brk_pt_start:brk_pt_stop2]) )
                # max_stop = np.where(np.diff(bin_rows[brk_pt_start:brk_pt_stop2]) == max_diff )[0][0] 
                
                if len(exact_brk_point) == 0: # None of the diff (first derivative) is greater than thresh, so just find the max derivative                    
                    exact_brk_point = np.where(np.diff(bin_rows[brk_pt_start:brk_pt_stop2]) == max_diff )[0] 
                    exact_brk_point = exact_brk_point[::-1] 
                
                ## Change exact_brk_point to a single value in case len(exact_brk_point) >1
                exact_brk_point = exact_brk_point[exact_brk_point>0][0].item() if len(exact_brk_point)>1 else  exact_brk_point.item() 
                
                ## Set the appropriate brk_pt_stop
                
                brk_pt_stop =  (exact_brk_point  + brk_pt_start ) if exact_brk_point <= Nx else (brk_pt_start  + Nx )
                brk_pt_start2, brk_pt_stop2 = None, None 
                                         
            else:
                # There are less than Nx bin_rows left - should be the last layer
                brk_pt_stop = len(bin_rows)   
                                             
            
            # NEED TO CONFIRM THIS
            # brk_pt_stop = brk_pt_stop2 - max_stop # brk_pt_stop = brk_pt_stop2 - Nx # This might not be correct          
            # brk_pt_start2 = brk_pt_stop if brk_pt_stop2 < len(bin_rows) else brk_pt_start

        if brk_pt_stop > brk_pt_start:            
            vec_layer = np.concatenate( (vec_layer,np.zeros(shape=(1,Nx)) ) )
            used_cols = bin_cols[brk_pt_start:brk_pt_stop+1] # Added extra 1 because of zero indexing
            used_rows = bin_rows[brk_pt_start:brk_pt_stop+1 ] # Added extra 1 because of zero indexing
            
            sort_idx = np.argsort(used_cols)
            used_cols = np.asarray( sorted(used_cols) )
            used_rows = used_rows[sort_idx]
            
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
        brk_pt_stop = brk_pt_start + Nx + 5 # Adding extra one to complete Nx and realizing Python indexing w/o last element
        
        brk_pt_stop2 = brk_pt_start2 = None           
         
        count +=1 
        
    return vec_layer



##==============================================================================##
## Create_vec_layer2
##==============================================================================##
# =============================================================================
def create_vec_layer2(raster,threshold={'constant':5}):
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
        vec_layer = create_vec_layer(res0_hard_threshold, 10)

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

##==============================================================================##
## # Path to data
##==============================================================================##    

base_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data'
season_name = 'snow_2012_Greenland_P3'   # <= e.g 'snow2_2021_Alaska_SO', 'snow_2021_Alaska_SO',  'snow_2012_Greenland_P3' , 'mcords2_2012_Greenland_P3'
#segments = ['20120330_02', '20120330_03', '20120404_01', '20120413_01', '20120418_01' , '20120429_01', '20120507_01', '20120508_01', '20120514_01'  ]
segments = ['20120413_01']   # ['20120413_01'] , ['20120418_01'],  ['20210509_01'] 

for segment in segments:
    
    ## Model path and files
    if decimated_model:       
        # model_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\new_trainJuly\SimpleUNet_Binary\SimpleUNet_acc_0.94_GOOD_16_July_22_1152.h5'
        model_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\new_trainJuly\ConvMixer\SegAcc_ 71.88_25_August_22_0937.h5'
        
        model_pred_data_path =  os.path.join(r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Frames_ML_pred',segment+'\*.mat')
        
    else:
        model_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data\SimpleUNet_Binary_large\SimpleUNet_acc_0.99_no_fixed_shape_13_October_22_0805.h5' 
        #model_pred_data_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Frames_ML_pred\20120330_04\full_size\*.mat' SimpleUNet_acc_0.99_no_fixed_shape_13_October_22_0805.h5, SimpleUNet_acc_0.99_GOOD_12_August_22_1544.h5
        
        model_pred_data_path =  os.path.join(r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Frames_ML_pred',season_name, segment+  r'\full_size\*.mat')
    
        
    ## Load saved model and weights
    model_custom_objects = {'iou_score':sm.metrics.iou_score}
    
    if not model_custom_objects:
        loaded_model = tf.keras.models.load_model(model_path) 
    else:
        loaded_model = tf.keras.models.load_model(model_path, custom_objects=model_custom_objects)
        
    model = loaded_model
    
    # Legacy file paths
    # model_pred_data_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\unlabeled_data\20120416_01\*.mat' #model_pred_data_path = os.path.join(base_path,r'unlabeled_data\20120516_01\*.mat')  #model_pred_data_path = os.path.join(base_path,r'new_trainJuly\new_test\2012*.mat')
    
    model_pred_data = sorted(glob.glob(model_pred_data_path)) 
    
    filter_x,filter_y = 11,51
    conv_filter = np.ones(shape=(filter_x,filter_y )) 
    

    ##==============================================================================##
    ## Visualize result of model prediction for "unseen" echogram during training
    ##==============================================================================##
    
    
    if binary_viz:    
        batch_idx = random.randint(0 ,len(model_pred_data) - 10) if len(model_pred_data) >=10 else 0  # Pick any of the default batch
        
        for idx in range(min(10,len(model_pred_data) ) ):
 
          predict_data = loadmat(model_pred_data[batch_idx+idx])
          #echo_tmp,a_gt0 = predict_data['echo_tmp'], predict_data['raster']
          echo_tmp = predict_data['echo_tmp']
          
          if model.input_shape[-1]  >1:
              a0 = np.stack((echo_tmp,)*3,axis=-1)
              res0 = model.predict ( np.expand_dims(a0,axis=0))
          else:
              res0 = model.predict ( np.expand_dims(np.expand_dims(echo_tmp,axis=0),axis=3) ) 
              
          res0 = res0.squeeze() # Probability map output of the model
          res0_hard_threshold = np.where(res0>0.1,1,0) # Threshold should be adaptive (based on pmap)        
                    

         # Remove islands and discontinuities in thresholded predictions
          res0_island_rmv = res0_hard_threshold.copy() ; 
          conv_vals = cv.filter2D(res0_island_rmv, -1, conv_filter, borderType=cv.BORDER_CONSTANT)          
          res0_island_rmv[conv_vals < np.max(conv_vals)//1.5] = 0 # Remove island predictions
          
          # Filter probability map??
          b = np.ones((15,))/15; a = 1    
          prob_map_filtered = filtfilt(b,a,res0.T).T 
          
          res_Nt,res_Nx = res0_hard_threshold.shape          

          res0_hard_threshold1 = np.arange(1,res_Nt+1).reshape(res_Nt,1) * fix_final_prediction(prob_map_filtered,res0_island_rmv,closeness=15)
          
          thresh = {'constant': 20} # This determines the minimum seperation between layers
          new_layer = create_vec_layer2(res0_hard_threshold1,thresh); 
          
          new_layer[new_layer==0] = np.nan;
          new_layer_filtered = new_layer.copy()          
          new_layer_filtered[:] = np.nan         

          for chan in range(new_layer.shape[0]):
              new_layer_curr = new_layer[chan,:]
              if ~np.all(np.isnan(new_layer_curr)) and len(new_layer_curr[~np.isnan(new_layer_curr)]) > 21:
                  # z_curr[np.isnan(z_curr)] = 0
                  new_layer_filtered[chan,:] =  sc_med_filt(new_layer_curr, size=35).astype('int32') #sc_med_filt(z,size=3)
              else:
                  new_layer_filtered[chan,:] = np.nan
          
          
          new_layer_filtered [ new_layer_filtered< 0] = np.nan
          
          # Finish this later
          # del_idx = np.sum( np.isnan(z_filtered).astype(int), axis = 1)
          # del_idx = (del_idx < res_Nx//3)         
          # z_filtered = np.delete(z_filtered,del_idx, 0) 
          
          f, axarr = plt.subplots(1,6,figsize=(20,20))
        
          axarr[0].imshow(echo_tmp.squeeze(),cmap='gray_r')
          axarr[0].set_title( f'Echo {os.path.basename(model_pred_data[batch_idx+idx])}') #.set_text
          
          axarr[1].imshow(echo_tmp.squeeze(),cmap='viridis')
          axarr[1].set_title( 'Echo orig map') #.set_text
          
          axarr[2].imshow(prob_map_filtered, cmap='viridis' )
          axarr[2].set_title('Prob_map_filtered') 
          
          axarr[3].imshow(res0_hard_threshold, cmap='viridis' )
          axarr[3].set_title('Thresholded Prediction [res0_hard_threshold]') 
          
          axarr[4].plot(new_layer.T) # gt
          axarr[4].invert_yaxis()
          axarr[4].set_title( f'Vec_layer({thresh})') #.set_text
          
          axarr[5].imshow(echo_tmp.squeeze(),cmap='gray_r')          
          axarr[5].plot(new_layer_filtered.T) # gt
          axarr[5].set_title( 'Overlaid prediction') #.set_text
          #axarr[2].set_title( f'Ground truth {os.path.basename(model_pred_data[batch_idx])}') #.set_text
          
          
        ##############################################################################  
    
    
    
    ##==============================================================================##
    ## Segmentation: Visualize result of model prediction for "unseen" echogram during training
    ##==============================================================================##
    
    if segmentation_viz:
        batch_idx = random.randint(1,len(model_pred_data) - 10) # Pick any of the default batch
        
        for idx in range(1,10):
          predict_data = loadmat(model_pred_data[batch_idx+idx])
          #echo_tmp,a_gt0 = predict_data['echo_tmp'], predict_data['raster']
          echo_tmp = predict_data['echo_tmp']
          
          if model.input_shape[-1] == 3:
              a0 = np.stack((echo_tmp,)*3,axis=-1)
              res0 = model.predict ( np.expand_dims(a0,axis=0))
          else:
              res0 = model.predict ( np.expand_dims(echo_tmp,axis=0) ) 
              
          res0 = res0.squeeze()
          res0_hard_threshold = np.argmax(res0,axis=2)
          #res0_hard_threshold = np.where(res0>0.1,1,0)
          
          res0_hard_threshold1 = sc_med_filt( sc_med_filt(res0_hard_threshold.T,size=7).T, size= 7)
          
    
        
          f, axarr = plt.subplots(1,3,figsize=(20,20))
        
          axarr[0].imshow(echo_tmp.squeeze(),cmap='gray_r')
          axarr[0].set_title( f'Echo {os.path.basename(model_pred_data[batch_idx])}') #.set_text
          
          axarr[1].imshow(res0_hard_threshold, cmap='viridis' )
          axarr[1].set_title('Prediction')
          
          axarr[2].imshow(res0_hard_threshold1, cmap='viridis' )
          axarr[2].set_title('Filtered Prediction')
    
    
##====================================================================================##
## Based on threshold from vizualization: Predict and save predictions for each echo
##====================================================================================##
    
    
    if run_predictions:
        # CHANGE THIS!!!
        
        for iter in range(len(model_pred_data)): #len(model_pred_data)
            # Load mat file
            predict_data = loadmat(model_pred_data[iter])
            
            # Load echo file
            echo_tmp =  predict_data['echo_tmp']
            
            # Check model type and predict    
            if model.input_shape[-1]  >1:
                a0 = np.stack((echo_tmp,)*3,axis=-1)
                res0 = model.predict ( np.expand_dims(a0,axis=0))
            else:
                res0 = model.predict ( np.expand_dims(np.expand_dims(a0,axis=0),axis=3) ) 
                
            res0 = res0.squeeze()
            
            # Low threshold
            res0_hard_threshold0 = np.where(res0>0.05,1,0)
            
            res0_hard_threshold = sc_med_filt( sc_med_filt(res0_hard_threshold0.T,size=3).T, size= 3)
            
            
            # Correct threshold and convert prediction to sparse matrix
            if decimated_model:
                res0_hard_threshold1 = np.arange(1,417).reshape(416,1) * fix_final_prediction(res0,res0_hard_threshold)
            else:
                res0_hard_threshold1 = np.arange(1,416*4+1).reshape(416*4,1) * fix_final_prediction(res0,res0_hard_threshold)
            
            # Convert sparse matrix to dense prediction (might have align issues: to be fixed in MATLAB)
            result ={}
            thresh = 8
            result['ML_layer'] =  create_vec_layer(res0_hard_threshold1, thresh)    
            
            
            # Save predictions    
            
            if save_predictions:
                dir_name = os.path.dirname(model_pred_data[iter])
                seg_name = os.path.basename(dir_name)
                folder_name = dir_name + '\predictions_' +  seg_name
                
                if not os.path.isdir(folder_name):
                    os.makedirs(folder_name)
                
                if '_dec' in os.path.basename(model_pred_data[iter]):
                    trunc_idx = os.path.basename(model_pred_data[iter]).index('_dec')
                    save_path = os.path.join( folder_name, os.path.basename(model_pred_data[iter])[:trunc_idx]+'.mat'  )
                else:
                    save_path = os.path.join( folder_name, os.path.basename(model_pred_data[iter]) )
                    
                
                savemat(save_path,result)    
    
################################################################################## 
    


##==============================================================================##
##  Extra functions for layer thickness estimation
##==============================================================================##

def zero_runs(a):  # from link
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


import itertools
def compute_layer_thickness(model,test_data):
    test_pred = model.predict(test_data)
    test_pred = np.argmax( test_pred, axis = 3)
    
    test_result = np.zeros((test_pred.shape[0],30,test_pred.shape[-1] ))
    ## Calculate thickness for each block
    
    for block_idx in range(test_pred.shape[0]):
        
        # First filter prediction
        temp1 = sc_med_filt( sc_med_filt(test_pred[block_idx],size=5).T, size=5, mode='nearest').T        
        
        for col_idx in range(temp1.shape[1]):
            uniq_layers = np.unique(temp1[:,col_idx])
            uniq_layers = uniq_layers[uniq_layers>0] # Zero is no layer class
            
            col_thickness = [ len(list(group)) for key, group in itertools.groupby(temp1[:,col_idx] ) if key ]
            
            for layer_idx in range(len(uniq_layers)):
                test_result[block_idx, uniq_layers[layer_idx], col_idx ] = col_thickness[layer_idx]
                
    test_result = np.transpose(test_result,axes= [1,0,2])
    test_result = test_result.reshape((30,-1),order='F')
    
    return test_result
                  
            
#layer_thickness = compute_layer_thickness(model, test_ds)               
#(np.mean(layer_thickness,axis =1)).round()

#############################################################################################  





















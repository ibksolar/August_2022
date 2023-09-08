# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 01:04:34 2022

@author: i368o351
"""

from __future__ import print_function

import os

import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
import tensorflow_addons as tfa

from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,ReduceLROnPlateau
from scipy.ndimage import median_filter as sc_med_filt

from scipy.io import loadmat
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import glob
# import cv2 as cv

import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()

print("Packages Loaded")


## GPU Config
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
#tf.keras.mixed_precision.set_global_policy('mixed_float16')


use_wandb = True
time_stamp = '23_April_23_1127' #datetime.strftime( datetime.now(),'%d_%B_%y_%H%M') #'10_April_23_2112' 

model_name ='res50_pretrained'


if use_wandb:    
    ## WandB config
    import wandb
    from wandb.keras import WandbCallback    
    
    wandb.init( project="my-test-project", entity="ibksolar", name = model_name+time_stamp,config ={})
    config = wandb.config
else:
    config ={}

#========================================================================
# ==================LOAD DATA =========================================
#========================================================================

# PATHS
# Path to data
base_path = r'U:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data'  # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'
train_path = os.path.join(base_path,'train_data\*.mat')
train_aug_path = os.path.join(base_path,'augmented_plus_train_data\*.mat')
val_path = os.path.join(base_path,'val_data\*.mat')
test_path = os.path.join(base_path,'test_data\*.mat')   

# Hyperparameters
config['Run_Note'] = ''
config['batch_size'] = 4


# Training params
config['img_y'] = 416*4
config['img_x'] = 64*4

config['img_channels'] = 1
config['weight_decay'] = 0.0001

config['num_classes'] = 1 #30
config['epochs'] = 150
config['learning_rate'] = 1e-3
config['base_path'] = base_path
SEED = 42
AUTO = tf.data.experimental.AUTOTUNE



# =============================================================================
# Function for training data
def read_mat_heavy_aug(filepath):
    def _read_mat(filepath):
        
        filepath = bytes.decode(filepath.numpy())      
        mat_file = loadmat(filepath)
        echo = tf.cast(mat_file['echo_tmp'], dtype=tf.float64)
        
        echo = tf.expand_dims(echo, axis=-1)
        
        if config['img_channels'] > 1:
            echo = tf.image.grayscale_to_rgb(echo)
        
        layer = tf.cast(mat_file['semantic_seg'], dtype=tf.float64)
        
        # Data Augmentation     
        
        # echo = tf.experimental.numpy.fliplr(echo)
        # layer = tf.experimental.numpy.fliplr(layer)
        
        if tf.random.uniform(())> 0.3:
            # Constant offset 1
            echo = echo - tf.random.normal(shape=(1,1),dtype=tf.float64)
            #echo = tf.clip_by_value(echo, 0, 1) 
            
            # # Random Add
            echo = echo + tf.random.normal(shape=(config['img_y'],config['img_x'],config['img_channels']),stddev=0.5,dtype=tf.float64)
            #echo = tf.clip_by_value(echo, 0, 1)  
            
            # Random contrast
            echo = tf.image.random_contrast(echo,0.2, 123)
            echo = tf.clip_by_value(echo, 0, 1) 
            
            # echo = tf.image.random_flip_left_right(echo)
            # layer = tf.image.random_flip_left_right(layer)
            
        else:            
        
            # Constant offset 2
            echo = echo + tf.random.normal(shape=(1,1),dtype=tf.float64)
            #echo = tf.clip_by_value(echo, 0, 1) 
            
            # # Random Subtract
            echo = echo - tf.random.normal(shape=(config['img_y'],config['img_x'],config['img_channels']),stddev=0.5,dtype=tf.float64)
            # #echo = tf.clip_by_value(echo, 0, 1)       
                
            # Random brightness
            echo = tf.image.random_brightness(echo,0.2, 123)
            #echo = tf.clip_by_value(echo, 0, 1)  
            
            # Random Saturation
            echo = tf.image.random_saturation(echo, 0.1, 0.9)
            echo = tf.clip_by_value(echo, 0, 1)
                
        layer = tf.keras.utils.to_categorical(layer, config['num_classes'] )
        shape0 = echo.shape #mat_file['echo_tmp'].shape
        
        return echo,layer,np.asarray(shape0)
    output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double, tf.int64])
    shape = output[2]
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([config['img_y'],config['img_x'],config['img_channels']])
    
    data1 = output[1]   
    data1.set_shape([config['img_y'],config['img_x'],30])#,30    
    return data0,data1


# Copied from EchoVIT1_large
# =============================================================================
# Function for training data
def read_mat_train(filepath):
    def _read_mat(filepath):
        
        filepath = bytes.decode(filepath.numpy())      
        mat_file = loadmat(filepath)
        echo = tf.cast(mat_file['echo_tmp'], dtype=tf.float64)
        
        echo = tf.expand_dims(echo, axis=-1)
        
        if config['img_channels'] > 1:
            echo = tf.image.grayscale_to_rgb(echo)
        
        # layer = tf.cast(mat_file['raster'], dtype=tf.float64)
        layer = tf.cast( tf.cast(mat_file['raster'], dtype=tf.bool), dtype=tf.float64)
        
        # Data Augmentation
        
        
        aug_type = tf.random.uniform((1,1),minval=1, maxval=8,dtype=tf.int64).numpy()
        
        if aug_type == 1:
            echo = tf.experimental.numpy.fliplr(echo)
            layer = tf.experimental.numpy.fliplr(layer)
        
        elif aug_type == 2: # Constant offset
            echo = echo - 0.3
        
        elif aug_type == 3: # Random noise
            echo = echo - tf.random.normal(shape=(1664,256,config['img_channels']),stddev=0.5,dtype=tf.float64)
            echo = tf.clip_by_value(echo, 0, 1)
            
        elif aug_type == 4: # Random brightness
            echo = tf.image.random_brightness(echo,0.2, 123)
            echo = tf.clip_by_value(echo, 0, 1)
            
        elif aug_type == 5: # Random contrast
            echo = tf.image.random_contrast(echo,0.2, 123)
            echo = tf.clip_by_value(echo, 0, 1)                
        
        elif aug_type == 6: # Random hue
            echo = tf.image.random_hue(echo,0.2, 123)
            echo = tf.clip_by_value(echo, 0, 1)
            
        elif aug_type == 7: # Random brightness
            echo = tf.image.random_saturation(echo, 0.1, 0.9)
            echo = tf.clip_by_value(echo, 0, 1)
        
        else: #aug_type == 4:
            echo = tf.experimental.numpy.flipud(echo)
            layer = tf.experimental.numpy.flipud(layer)                            
                   
        layer = np.expand_dims(layer, axis=-1)
        #layer = tf.keras.utils.to_categorical(layer, config['num_classes'] )
        shape0 = echo.shape #mat_file['echo_tmp'].shape
        
        return echo,layer,np.asarray(shape0)
    output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double, tf.int64])
    shape = output[2]
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([config['img_y'],config['img_x'],config['img_channels']])

    
    data1 = output[1]   
    data1.set_shape([config['img_y'],config['img_x'],1])
    # data1.set_shape([1664,256,1 ])#,30,config['num_classes']    
    return data0,data1

# =============================================================================
## Function for test and validation dataset    
def read_mat(filepath):
    def _read_mat(filepath):
        
        filepath = bytes.decode(filepath.numpy())      
        mat_file = loadmat(filepath)
        echo = tf.cast(mat_file['echo_tmp'], dtype=tf.float64)
        
        echo = tf.expand_dims(echo, axis=-1)        
        if config['img_channels'] > 1:
            echo = tf.image.grayscale_to_rgb(echo)
        
        # layer = tf.cast(mat_file['raster'], dtype=tf.float64)      
        layer = tf.cast( tf.cast(mat_file['raster'], dtype=tf.bool), dtype=tf.float64)
        
        layer = tf.expand_dims(layer, axis=-1)
        # layer = tf.keras.utils.to_categorical(layer, config['num_classes'] )
        shape0 = echo.shape #mat_file['echo_tmp'].shape        
        return echo,layer,np.asarray(shape0)     
    
    output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double, tf.int64])
    shape = output[2]
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([config['img_y'],config['img_x'],config['img_channels']])
    
    data1 = output[1]   
    data1.set_shape([config['img_y'],config['img_x'],1]) #,30   
    return data0,data1

train_ds = tf.data.Dataset.list_files(train_aug_path,shuffle=True) #'*.mat'
train_ds = train_ds.map(read_mat,num_parallel_calls=8)
train_ds = train_ds.batch(config['batch_size'],drop_remainder=True).prefetch(AUTO) #.shuffle(buffer_size = 100 * config['batch_size'])

# No augmentation for testing and validation
val_ds = tf.data.Dataset.list_files(val_path,shuffle=True)
val_ds = val_ds.map(read_mat,num_parallel_calls=8)
val_ds = val_ds.batch(config['batch_size'],drop_remainder=True).cache().prefetch(AUTO)

test_ds = tf.data.Dataset.list_files(test_path,shuffle=True)
test_ds = test_ds.map(read_mat,num_parallel_calls=8)
test_ds = test_ds.batch(config['batch_size'],drop_remainder=True).cache().prefetch(AUTO)

train_shape = [ ( tf.shape(item[0]).numpy(),tf.shape(item[1]).numpy() ) for item in train_ds.take(1) ]
train_shape = train_shape[0]

print(f' X_train train shape {train_shape[0]}')
print(f' Training target shape {train_shape[1]}')



# Using segmentation-models as base



input_shape = (None,None,1)
inputs = layers.Input(shape=input_shape,name='Custom_input1')

resnet_50 = sm.Unet(backbone_name='resnet50', encoder_weights='imagenet',  encoder_freeze=True)
resnet_34 = sm.Unet(backbone_name='resnet34', encoder_weights='imagenet',  encoder_freeze=True)
fpn = sm.Unet(backbone_name='resnet152', encoder_weights='imagenet', encoder_freeze=True)

in1 = layers.Conv2D(3, 3,  padding="same", name = 'R1')(inputs)
in2 = layers.Conv2D(3, 3,  padding="same", name = 'R2')(inputs)
in3 = layers.Conv2D(3, 3,  padding="same", name = 'R3')(inputs)

output_res50 = resnet_50(in1)
output_res34 = resnet_34(in2)
output_fpn = fpn(in3)

combined = output_res34 + output_res50 + output_fpn

output_res50 = layers.Conv2D(64, 3, padding='same', activation='relu', name='CR1')(output_res50)
output_res50 = layers.Conv2D(1, 3, padding='same', activation='sigmoid', name='CR2')(output_res50)

output_res34 = layers.Conv2D(64, 3, padding='same', activation='relu', name='CR3')(output_res34)
output_res34 = layers.Conv2D(1, 3, padding='same', activation='sigmoid', name='CR4')(output_res34)

output_fpn = layers.Conv2D(64, 3, padding='same', activation='relu', name='CR5')(output_fpn)
output_fpn = layers.Conv2D(1, 3, padding='same', activation='sigmoid', name='CR6')(output_fpn)

combined = layers.Conv2D(64, 3, padding='same', activation='relu', name='CR7')(combined)
combined = layers.Conv2D(1, 3, padding='same', activation='sigmoid', name='CR8')(combined)

# layers.Conv2D(1, 3, activation="softmax", padding="same", dtype=tf.float64)(in2)
outputs = [output_res50, output_res34, output_fpn, combined]

model = Model(inputs=inputs, outputs=outputs, name='Res50_Res34_CR9_FPN')

# Compile and train
opt = tfa.optimizers.AdamW(learning_rate=config['learning_rate'], weight_decay = config['weight_decay'])
model.compile(optimizer= opt, loss="binary_crossentropy", #sparse_ categorical,"binary_crossentropy"
        metrics=["accuracy"], #,sm.metrics.iou_score
    )


config['start_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
logz= f"{config['base_path']}/{model_name}/{config['start_time']}_logs/"

checkpoint_filepath = f"{config['base_path']}//{model_name}//{model_name}_model_Checkpoint_new{config['start_time']}"

callbacks = [
   ModelCheckpoint(checkpoint_filepath, save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=0.000005, verbose= 1),
    EarlyStopping(monitor="val_loss", patience=30, verbose=1), 
    # TensorBoard(log_dir = logz,histogram_freq = 1,profile_batch = '1,70', embeddings_freq=50),
    WandbCallback()
]


history = model.fit(train_ds, epochs = config['epochs'],validation_data = val_ds, callbacks = callbacks)

#model.load_weights(checkpoint_filepath)

_, accuracy = model.evaluate(test_ds)
print(f"Test accuracy: {round(accuracy * 100, 2)}%")

model_save_path = checkpoint_filepath #f'{base_path}/{model_name}/{model_name}_model_{accuracy*100:.2f}_{config["start_time"]}.h5'
model.save(model_save_path)


from itertools import groupby
    
def fix_final_prediction(a, a_final,closeness = 15):
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
            
            close_locs_idx = np.where(np.diff(one_locs, prepend = 0) < closeness)[0]                
            
            for item in close_locs_idx:
                # Compare the probs of the "1" before the close 
                check1, check2 = one_locs[item-1]-1, one_locs[item]-1 # Indexing is off by one
                min_chk = check1 if a[check1,col_idx] < a[check2,col_idx] else check2
                a_final[min_chk, col_idx] = 0
        
    
    return a_final
##############################################################################    



def create_vec_layer(raster,threshold={'constant':25}, debug=False):
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
        
        # To be removed after debugging
        if debug:
            brk_pt_stop = brk_pt_stop if brk_pt_stop!=len(bin_rows) else brk_pt_stop-1
                   
            print('=================================================================================================')
            print(f'Current layer have {brk_pt_stop - brk_pt_start} elements')
            print(f'Layer {count}: Ends with { bin_rows[brk_pt_stop]} ({brk_pt_stop}), next layer starts with {bin_rows[brk_pt_stop+1]} {(brk_pt_stop+1)}')
            print(f'layer jump is {bin_rows[brk_pt_stop+1]} - {bin_rows[brk_pt_stop] } = {bin_rows[brk_pt_stop+1] - bin_rows[brk_pt_stop] } ')
            print('=================================================================================================')
        
        
        brk_pt_start = brk_pt_stop + 1
        brk_pt_stop = brk_pt_start + Nx + 5 # Adding extra one to complete 64 and realizing Python indexing w/o last element
        
        brk_pt_stop2 = brk_pt_start2 = None           
         
        count +=1


    # Legacy scripts: Should be deleted after harvesting
    #threshold = 7;     #brk_points = [ idx for (idx,value) in enumerate(np.diff(bin_rows)) if value > threshold ]   #np.diff(bin_rows)
    #brk_pt_chk = [ (idx,bin_rows[idx],value) for (idx,value) in enumerate(np.diff(bin_rows)) if value > threshold] 
    #brk_points = [-1] + brk_points + [len(bin_rows)]        
    # brk_points[0] = -1
    # num_layers = len(brk_points) - 1        
    # vec_layer = np.zeros( ( num_layers, raster.shape[-1] ) )   
    # for iter in range(num_layers):
    #     start_idx , stop_idx = brk_points[iter]+1 , brk_points[iter+1] + 1            
    #     vec_layer[iter, bin_cols[start_idx:stop_idx] ] = bin_rows[start_idx:stop_idx ]   
        
        
        
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




filter_x,filter_y = 11,51
conv_filter = np.ones(shape=(filter_x,filter_y ))


## Visualize result of model prediction for "unseen" echogram during training
model_val_data_path = os.path.join(base_path,'test_data\*.mat')
#model_val_data_path = os.path.join(base_path,'new_test\image\*.mat')
model_val_data = glob.glob(model_val_data_path)

batch_idx = random.randint(1,len(model_val_data)-10) # Pick any of the default batch

for idx in range(10):
  predict_data = loadmat(model_val_data[batch_idx+idx])
  a01,a_gt0 = predict_data['echo_tmp'], predict_data['raster']
  
  if config['img_channels'] > 1:
    a0 = np.stack((a01,)*3,axis=-1)
    res0 = model.predict ( np.expand_dims(a0,axis=0))
  else:
    res0 = model.predict ( np.expand_dims(np.expand_dims(a01,axis=0),axis=3) )  

  res0 = res0.squeeze()
  res0_final = np.where(res0>0.03,1,0)
  
  # Remove islands and discontinuities in thresholded predictions
  res0_island_rmv = res0_final.copy() ; 
  conv_vals = cv.filter2D(res0_island_rmv, -1, conv_filter, borderType=cv.BORDER_CONSTANT)          
  res0_island_rmv[conv_vals < np.max(conv_vals)//3] = 0 # Remove island predictions
  
  
  sz1 = res0_final.shape[0]
  
  if config['img_y']< 500:
      res0_final1 = np.arange(1,sz1+1).reshape(sz1,1) * fix_final_prediction(res0,res0_island_rmv,closeness =0)
  else:
      res0_final1 = np.arange(1,416*4+1).reshape(416*4,1) * fix_final_prediction(res0,res0_island_rmv,closeness =5)   
  
  # = np.ceil( filtfilt(b,a,res0_final1, axis =-1) ) 
   # How correct is create_vec_layer??
  thresh = {'constant': 30} # Maximum jump allowed in one layer
  new_layer = create_vec_layer(res0_final1,thresh)
   
  new_layer[new_layer==0] = np.nan;
  new_layer_filtered = new_layer.copy()          
  new_layer_filtered[:] = np.nan         
    
  for chan in range(new_layer.shape[0]):
      new_layer_curr = new_layer[chan,:]
      if ~np.all(np.isnan(new_layer_curr)) and len(new_layer_curr[~np.isnan(new_layer_curr)]) > 21:
          new_layer_filtered[chan,:] =  sc_med_filt(new_layer_curr, size=35).astype('int32') #sc_med_filt(z,size=3)
      else:
          new_layer_filtered[chan,:] = np.nan
          
          
  new_layer_filtered [ new_layer_filtered< 0] = np.nan
  
   #b = (np.ones((7,1))/7).squeeze(); a = 1;          
   #z_filtered =  filtfilt(b,a,z).astype('int32') #sc_med_filt(z,size=3)
   
  f, axarr = plt.subplots(1,6,figsize=(20,20))
     
  axarr[0].imshow(a01.squeeze(),cmap='gray_r')
  axarr[0].set_title( f'Echo {os.path.basename(model_val_data[batch_idx+idx])}') #.set_text
  
  axarr[1].imshow(a01.squeeze(),cmap='viridis')
  axarr[1].set_title( 'Echo orig map') #.set_text
  
  axarr[2].imshow(res0,cmap='viridis')
  axarr[2].set_title( 'Model direct output') #.set_text
   
  axarr[3].imshow(res0_final, cmap='viridis' ) #.astype(bool).astype(int)
  axarr[3].set_title('Thresholded prediction') 
   
  axarr[4].imshow(res0_final1.astype(bool).astype(int), cmap='viridis' )
  axarr[4].set_title('Thinned Prediction') 
   
  # axarr[3].plot(z.T) # gt
  # axarr[3].invert_yaxis()
  # axarr[3].set_title( f'Vec_layer({thresh})') #.set_text
   
  axarr[5].imshow(a01.squeeze(),cmap='gray_r')          
  axarr[5].plot(new_layer_filtered.T) # gt
  axarr[5].set_title('Overlaid prediction') #.set_text
  
  # axarr[5].imshow(a_gt0.astype(bool).astype(int), cmap='viridis' )
  # axarr[5].set_title('Ground truth') 
  

































# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 15:55:12 2021

Train and test using newly created Train and Test set

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

import segmentation_models as sm

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


model_name = 'ConvMixer_dec2raster_decimated_2stage'

use_wandb = True
time_stamp = '25_April_23_2303' #datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')

if use_wandb:    
    ## WandB config
    import wandb
    from wandb.keras import WandbCallback    
    
    wandb.init( project="my-test-project", entity="ibksolar", name= model_name+time_stamp,config ={})
    config = wandb.config
else:
    config ={}




#========================================================================
# ==================LOAD DATA =========================================
#========================================================================

# PATHS
# Path to data
base_path = r'U:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\SR_Dataset_v1\Dec' # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'
train_path = os.path.join(base_path,'train_data\*.mat')
# train_aug_path = os.path.join(base_path,'augmented_plus_train_data\*.mat')
val_path = os.path.join(base_path,'val_data\*.mat')
test_path = os.path.join(base_path,'test_data\*.mat')   

# Hyperparameters
config['Run_Note'] = 'ConvMixer_dec2raster_decimated_2stage'
config['batch_size'] = 8


# Training params
config['num_layers'] =  32

config['img_y'] = 416 #*4
config['img_x'] = 64*4

config['img_channels'] = 1
config['weight_decay'] = 0.0001

config['num_classes'] = 1 #30
config['epochs'] = 100
config['learning_rate'] = 1e-3
config['base_path'] = base_path
SEED = 42
AUTO = tf.data.experimental.AUTOTUNE




# =============================================================================
## Function for multi-input reading for Convolution and Regression

def raster_read_mat(filepath):
    def _read_mat(filepath):
        
        dtype = tf.float64
        
        filepath = bytes.decode(filepath.numpy())      
        mat_file = loadmat(filepath)
        # echo = tf.cast(mat_file['echo_tmp'], dtype= dtype) #, dtype=tf.float64        
        # echo = tf.expand_dims(echo, axis=-1)
        
        echo = tf.cast( tf.cast(mat_file['new_raster'], dtype=tf.bool), dtype = dtype)
        echo = tf.expand_dims(echo, axis=-1)
        
        layer = tf.cast(mat_file['regress_32'], dtype = dtype)         
        layer = tf.expand_dims(layer, axis=-1)
        
        shape0 = echo.shape        
        return echo,layer,np.asarray(shape0)     
    
    output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double,tf.int64]) #,tf.double, tf.half
    shape = output[2]
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([config['img_y'],config['img_x'], config['img_channels'] ])
    
    data1 = output[1]  
    # data1.set_shape([config['img_y'],config['img_x'], config['img_channels'] ])  #,30, ,config['num_classes'] 
    data1.set_shape([config['img_y']//13,config['img_x'], 1])

    
    return data0,data1

train_ds = tf.data.Dataset.list_files(train_path,shuffle=True) #'*.mat'
train_ds = train_ds.map(raster_read_mat,num_parallel_calls=8)
train_ds = train_ds.batch(config['batch_size'],drop_remainder=True).prefetch(AUTO) #.shuffle(buffer_size = 100 * config['batch_size'])

# No augmentation for testing and validation
val_ds = tf.data.Dataset.list_files(val_path,shuffle=True)
val_ds = val_ds.map(raster_read_mat,num_parallel_calls=8)
val_ds = val_ds.batch(config['batch_size'],drop_remainder=True).cache().prefetch(AUTO)

test_ds = tf.data.Dataset.list_files(test_path,shuffle=True)
test_ds = test_ds.map(raster_read_mat,num_parallel_calls=8)
test_ds = test_ds.batch(config['batch_size'],drop_remainder=True).cache().prefetch(AUTO)

# train_shape = [ ( tf.shape(item[0]).numpy(),tf.shape(item[1]).numpy() ) for item in train_ds.take(1) ]
# train_shape = train_shape[0]
# print(f' X_train train shape {train_shape[0]}')
# print(f' Training target shape {train_shape[1]}')



# input_shape = (config['img_y'], config['img_x'], config['img_channels'])
input_shape = (None, None, config['img_channels'])

# data_augmentation = tf.keras.Sequential(
#     [
#         layers.experimental.preprocessing.Rescaling(scale=1.0 / 255),
#         layers.experimental.preprocessing.RandomCrop(image_x, image_y),
#         layers.experimental.preprocessing.RandomFlip("vertical"),
#     ],
#     name="data_augmentation",
# )
#========================================================================
# ================== MODEL=========================================
#========================================================================

def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def conv_stem(x, filters: int, patch_size: int): #, patch_size: int
    x = layers.Conv2D(filters, kernel_size=patch_size,padding="same")(x) #, strides=patch_size
    # x = layers.Conv2D(filters, kernel_size=(image_x,image_y) )(x)
    return activation_block(x)


def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block(x), x0])  # Residual.

    # Pointwise convolution.
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)

    return x


def get_conv_mixer_256_8(
     filters=64, depth=2, kernel_size=(17,15), patch_size=(1,config['img_y']), num_classes=config['num_classes']): #depth=8, kernel_size=5  patch_size=2,
    """ConvMixer-256/8: https://openreview.net/pdf?id=TVHS5Y4dNvM.
    The hyperparameter values are taken from the paper.
    """
    inputs = tf.keras.Input(input_shape)
    # x = layers.Rescaling(scale=1.0 / 255)(inputs)

    # Extract patch embeddings.
    x = conv_stem(inputs, filters, patch_size) #, patch_size

    # ConvMixer blocks.
    for _ in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)

    # DepthWise Classification block.
    x = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(x)
    # x = layers.GlobalAvgPool2D()(x)
    x= layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', strides=(2, 2), padding='same')(x) #,strides=(2, 2)
    x = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    x = layers.Dropout(0.4)(x)
    # bin_output = layers.Conv2D(config['num_classes'],(1,1), padding = 'same',activation="sigmoid", dtype=tf.float64 , name= "raster" )(x) #softmax, sigmoid, activation="softmax", 
        
    combined_out = layers.Conv2D(256, (17,15), activation='relu', padding="same") (x)
    combined_out = layers.Conv2D(128, 5, activation='relu', padding="same") (combined_out)    
    combined_out = layers.MaxPool2D(pool_size=(13,1)) (combined_out)
    
    combined_out = layers.Conv2D(128, 3, activation='relu', padding="same") (combined_out)
    combined_out = layers.Conv2D(64, 3, activation='relu', padding="same") (combined_out)  
    combined_out = layers.Dropout(0.3) (combined_out)
    combined_out = layers.Conv2D( 1, (1,1), padding="same",activation="sigmoid", dtype = tf.float64)(combined_out) # activation='sigmoid', name= "raster3D", 
    
    # combined_out = layers.GlobalAverage()(combined_out)    
    # combined_out = layers.Dense(config['num_layers'] * config['img_x'], activation="sigmoid", dtype = tf.float64)(combined_out)
    
    
    return Model(inputs, outputs = combined_out)



#========================================================================
# =============== MODEL TRAINING AND EVALUATION =========================
#========================================================================

#loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
 

def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=config['learning_rate'], weight_decay=config['weight_decay']
    )

    # model.compile(
    #     optimizer=optimizer,
    #     loss="binary_crossentropy", #sparse_ categorical,"binary_crossentropy"
    #     metrics=["accuracy"]#,sm.metrics.iou_score],
    model.compile(optimizer=optimizer,#opt
              loss =  "mean_squared_error", #, "mean_squared_error", "mean_squared_error", "mean_squared_error" ,              
              metrics=[ ['mean_squared_error' ,'mean_absolute_error']  ] )
    
    

    checkpoint_filepath = os.path.abspath(base_path+f"/{model_name}/{time_stamp}_checkpoint.h5")
    checkpoint_callback = [
        ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        #save_weights_only=True,
    ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.25, patience=10, min_lr=0.00005, verbose= 1),
        EarlyStopping(monitor="val_loss", patience=30, verbose=1)  ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config['epochs'],
        callbacks=[checkpoint_callback, WandbCallback() ], #WandbCallback()
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy = model.evaluate(test_ds)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, model


conv_mixer_model = get_conv_mixer_256_8()

time_stamp = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
print(f"Start time: {time_stamp}")

history, conv_mixer_model = run_experiment(conv_mixer_model)
time_stamp = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')

print(f"End time: {time_stamp}")

_, accuracy = conv_mixer_model.evaluate(test_ds)

model_save_path = f'{base_path}/{model_name}/SegAcc_{accuracy*100:.2f}_{time_stamp}.h5'

conv_mixer_model.save(model_save_path)




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



## Visualize result of model prediction for "unseen" echogram during training
model_val_data_path = os.path.join(base_path,'test_data\*.mat')
#model_val_data_path = os.path.join(base_path,'new_test\image\*.mat')
model_val_data = glob.glob(model_val_data_path)

batch_idx = random.randint(1,len(model_val_data)-10) # Pick any of the default batch
model = conv_mixer_model

for idx in range(10):
  predict_data = loadmat(model_val_data[batch_idx+idx])
  a01,a_gt0 = predict_data['echo_tmp'], predict_data['raster']
  
  if config['img_channels'] > 1:
    a0 = np.stack((a01,)*3,axis=-1)
    res0 = model.predict ( np.expand_dims(a0,axis=0))
  else:
    res0 = model.predict ( np.expand_dims(np.expand_dims(a01,axis=0),axis=3) )  

  res0 = res0.squeeze()
  res0_final = np.where(res0>0.01,1,0)
  
  sz1 = res0_final.shape[0]
  
  if config['img_y']< 500:
      res0_final1 = np.arange(1,sz1+1).reshape(sz1,1) * fix_final_prediction(res0,res0_final)
  else:
      res0_final1 = np.arange(1,416*4+1).reshape(416*4,1) * fix_final_prediction(res0,res0_final)   
  
  # = np.ceil( filtfilt(b,a,res0_final1, axis =-1) ) 
   # How correct is create_vec_layer??
  thresh = {'constant': 15}
  z = create_vec_layer(res0_final1,thresh); 
   
  z[z==0] = np.nan;
   
   #b = (np.ones((7,1))/7).squeeze(); a = 1;          
   #z_filtered =  filtfilt(b,a,z).astype('int32') #sc_med_filt(z,size=3)
   
  f, axarr = plt.subplots(1,6,figsize=(20,20))
     
  axarr[0].imshow(a01.squeeze(),cmap='gray_r')
  axarr[0].set_title( f'Echo {os.path.basename(model_val_data[batch_idx+idx])}') #.set_text
   
  axarr[1].imshow(res0_final.astype(bool).astype(int), cmap='viridis' )
  axarr[1].set_title('Prediction before threshold') 
   
  axarr[2].imshow(res0_final1.astype(bool).astype(int), cmap='viridis' )
  axarr[2].set_title('Prediction') 
   
  axarr[3].plot(z.T) # gt
  axarr[3].invert_yaxis()
  axarr[3].set_title( f'Vec_layer({thresh})') #.set_text
   
  axarr[4].imshow(a01.squeeze(),cmap='gray_r')          
  axarr[4].plot(z.T) # gt
  axarr[4].set_title( 'Overlaid prediction') #.set_text
  
  axarr[5].imshow(a_gt0.astype(bool).astype(int), cmap='viridis' )
  axarr[5].set_title('Ground truth') 
  
  
  

      

     

       
   








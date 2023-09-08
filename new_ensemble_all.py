# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 16:26:21 2022

@author: i368o351
"""

from tensorflow.keras import layers
from tensorflow import keras
from keras import backend as K

import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm,colors

import os
import random
from scipy.io import loadmat
from scipy.ndimage import median_filter as sc_med_filt



import glob
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,ReduceLROnPlateau
import segmentation_models as sm
from datetime import datetime



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
tf.keras.mixed_precision.set_global_policy('mixed_float16')


use_wandb = True
time_stamp = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')

if use_wandb:    
    ## WandB config
    import wandb
    from wandb.keras import WandbCallback    
    
    wandb.init( project="my-test-project", entity="ibksolar", name='New_Ensemble'+time_stamp,config ={})
    config = wandb.config
else:
    config ={}
    
    
    
#========================================================================
# ==================LOAD DATA =========================================
#========================================================================
# PATHS
# Path to data
base_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data'  # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'
train_path = os.path.join(base_path,'train_data\*.mat')
train_aug_path = os.path.join(base_path,'augmented_plus_train_data\*.mat')
val_path = os.path.join(base_path,'val_data\*.mat')
test_path = os.path.join(base_path,'test_data\*.mat')   

# Hyperparameters
config['Run_Note'] = 'New combined_ensemble'
config['batch_size'] = 4


# Training params
config['img_y'], config['img_x'] = loadmat(glob.glob(train_path)[0])['echo_tmp'].shape

config['img_channels'] = 3
config['weight_decay'] = 0.0001

config['num_classes'] = 1 #30
config['epochs'] = 500
config['learning_rate'] = 1e-3
config['base_path'] = base_path
SEED = 42
AUTO = tf.data.experimental.AUTOTUNE



# =============================================================================
## Function for dataset    
def read_mat(filepath):
    def _read_mat(filepath):
        
        filepath = bytes.decode(filepath.numpy())      
        mat_file = loadmat(filepath)
        echo = tf.cast(mat_file['echo_tmp'], dtype=tf.float64)
        
        echo = tf.expand_dims(echo, axis=-1)
        
        if config['img_channels']> 1:            
            echo = tf.image.grayscale_to_rgb(echo)          
        
        if config['num_classes'] == 1:
            layer = tf.cast( tf.cast(mat_file['raster'], dtype=tf.bool), dtype=tf.float64)        
            layer = tf.expand_dims(layer, axis=-1)
        else:
            layer = tf.cast(mat_file['semantic_seg2'], dtype=tf.float64) 
            layer = tf.expand_dims(layer, axis=-1)
            #layer = tf.keras.utils.to_categorical(layer, config['num_classes'] )
        
        shape0 = echo.shape        
        return echo,layer,np.asarray(shape0)     
    
    output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double, tf.int64])
    shape = output[2]
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([config['img_y'],config['img_x'], config['img_channels'] ])
    
    data1 = output[1]   
    data1.set_shape([config['img_y'],config['img_x'],config['num_classes'] ]) #,30, ,config['num_classes']    
    return data0,data1

train_ds = tf.data.Dataset.list_files(train_path,shuffle=True) #'*.mat'
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




#==============================================================================
# PatchEncoder and others
#==============================================================================
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches,  embed_dim,  **kwargs):
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



custom_objects ={'PatchEncoder':PatchEncoder, 'binary_focal_loss':sm.losses.BinaryFocalLoss(), 'iou_score':sm.metrics.iou_score, 'precision':sm.metrics.precision} 


# (0.) FCN 
FCN_path = os.path.join(base_path,r'SimpleFCNet\SimpleFCNet_Checkpoint22_November_22_2221.h5')
FCN = tf.keras.models.load_model(FCN_path, custom_objects=custom_objects)
FCN_in_shape = FCN.input_shape

#(1) SimpleUNet
SimpleUNet_path = os.path.join(base_path,r'SimpleUNet_Binary_large\SimpleUNet_acc_0.99_no_fixed_shape_13_October_22_0805.h5')
SimpleUNet = tf.keras.models.load_model(SimpleUNet_path)
SimpleUNet_in_shape = SimpleUNet.input_shape

# (2.) AttentionUNet
AttUNet_path = os.path.join(base_path,r'AttUNet\AttUNet_Checkpoint14_October_22_0940.h5')
AttUNet = tf.keras.models.load_model(AttUNet_path, custom_objects = custom_objects)
AttUNet_in_shape = AttUNet.input_shape

# (3.) DeepLab
DeepLab_path = os.path.join(base_path,r'DeepLab_binary_large\DeepLab_Checkpoint17_November_22_1149.h5')
DeepLab = tf.keras.models.load_model(DeepLab_path, custom_objects = custom_objects)
DeepLab_in_shape = DeepLab.input_shape

# (4.) Res50_pretrained ( This can take any input shape)
Res50_pretrained_path = os.path.join(base_path,'Res50_pretrained_model\Res50_pretrained_model_98.51_02_November_22_1845.h5')
Res50_pretrained = tf.keras.models.load_model(Res50_pretrained_path, custom_objects = custom_objects)
Res50_pretrained_in_shape = Res50_pretrained.input_shape


# all_models = [ FCN, SimpleUNet, AttUNet, DeepLab, Res50_pretrained ]
# for model in all_models:
#     for layer in model.layers:
#         layer.trainable = False



## MODEL
dtype_used = tf.float64
input_shape = (config['img_y'], config['img_x'], config['img_channels']  )

inputs = layers.Input(shape=input_shape, dtype=dtype_used)

inputs_1chan = layers.Conv2D(1, 3, padding = "same",activation='relu')(inputs)

FCN_x = tf.cast( FCN(inputs), dtype=dtype_used)
SimpleUNet_x = tf.cast( SimpleUNet(inputs_1chan) , dtype=dtype_used)
AttUNet_x = tf.cast( AttUNet(inputs) , dtype=dtype_used)
DeepLab_x = tf.cast(  DeepLab(inputs) , dtype=dtype_used)
Res50_x =tf.cast(  Res50_pretrained(inputs_1chan) , dtype=dtype_used)

output = FCN_x + SimpleUNet_x + AttUNet_x + DeepLab_x + Res50_x
output = layers.Conv2D(32,3,padding = 'same', activation = 'relu')(output)
output = layers.Conv2D(1,1,padding = 'same', activation = 'relu')(output)

model = keras.Model(inputs=inputs, outputs=output)

## Print Start time
config['start_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
print(f' Start time {config["start_time"]}')

logz= f"{config['base_path']}/new_ensemble/{config['start_time']}_logs/"
callbacks = [
   ModelCheckpoint(f"{config['base_path']}//new_ensemble//new_ensemble_Checkpoint{time_stamp}.h5", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.25, patience=10, min_lr=0.0005, verbose= 1),
    EarlyStopping(monitor="val_loss", patience=30, verbose=1), 
    TensorBoard(log_dir = logz,histogram_freq = 1,profile_batch = '1,70', embeddings_freq=50),
    WandbCallback()
]

model.compile(optimizer = 'Adam', loss= 'binary_crossentropy', metrics='accuracy')
history = model.fit(train_ds, epochs = config['epochs'],validation_data = val_ds, callbacks = callbacks)


## Print End time
config['start_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
print(f' Completion time {config["start_time"]}')



# Save model with proper name
_,acc = model.evaluate(test_ds)
model.save(f"{config['base_path']}//new_ensemble//new_ensemble_{acc:.2f}_{time_stamp}.h5")




















































































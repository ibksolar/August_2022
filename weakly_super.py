# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 14:18:36 2022

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


use_wandb = False
time_stamp = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')

if use_wandb:    
    ## WandB config
    import wandb
    from wandb.keras import WandbCallback    
    
    wandb.init( project="my-test-project", entity="ibksolar", name='Res50_Pretrained'+time_stamp,config ={})
    config = wandb.config
else:
    config ={}

try:
    fname = ipynbname.name()
except:
    fname = os.path.splitext( os.path.basename(__file__) )[0]
finally:
    print ('Could not automatically find file path')
    fname = 'blank'

#========================================================================
# ==================LOAD DATA =========================================
#========================================================================

# load base model
base_model = tf.keras.models.load_model(r'Y:\\ibikunle\\Python_Project\\Fall_2021\\all_block_data\\Attention_Train_data\\Full_size_data/Res50_pretrained_model/Res50_pretrained_model_98.51_25_October_22_1633.h5', compile=False)

# PATHS
# Path to data
base_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data'  # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'
train_path = os.path.join(base_path,'train_data\*.mat')
train_aug_path = os.path.join(base_path,'augmented_plus_train_data\*.mat')
val_path = os.path.join(base_path,'val_data\*.mat')
test_path = os.path.join(base_path,'test_data\*.mat')   

# Hyperparameters

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
# Function for training data with weak labels
    
def read_mat_weak_labels(filepath):
    def _read_mat(filepath):
        
        filepath = bytes.decode(filepath.numpy())      
        mat_file = loadmat(filepath)
        echo = tf.cast(mat_file['echo_tmp'], dtype=tf.float64)
        
        echo = tf.expand_dims(echo, axis=-1)  
        
        layer = base_model.predict ( np.expand_dims(np.expand_dims(echo,axis=0),axis=3) ) 
        layer = tf.cast(layer,dtype=tf.float64)
        layer = tf.squeeze(layer, axis=0)
        
   
        #layer = tf.cast( tf.cast(mat_file['raster'], dtype=tf.bool), dtype=tf.float64)        
        #layer = tf.expand_dims(layer, axis=-1)
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
train_ds = train_ds.map(read_mat_weak_labels,num_parallel_calls=8)
train_ds = train_ds.batch(config['batch_size'],drop_remainder=True).prefetch(AUTO) #.shuffle(buffer_size = 100 * config['batch_size'])

# No augmentation for testing and validation
val_ds = tf.data.Dataset.list_files(val_path,shuffle=True)
val_ds = val_ds.map(read_mat_weak_labels,num_parallel_calls=8)
val_ds = val_ds.batch(config['batch_size'],drop_remainder=True).cache().prefetch(AUTO)

test_ds = tf.data.Dataset.list_files(test_path,shuffle=True)
test_ds = test_ds.map(read_mat_weak_labels,num_parallel_calls=8)
test_ds = test_ds.batch(config['batch_size'],drop_remainder=True).cache().prefetch(AUTO)

train_shape = [ ( tf.shape(item[0]).numpy(),tf.shape(item[1]).numpy() ) for item in train_ds.take(1) ]
train_shape = train_shape[0]

print(f' X_train train shape {train_shape[0]}')
print(f' Training target shape {train_shape[1]}')



# Using segmentation-models as base

resnet_50 = sm.Unet(backbone_name='resnet50', encoder_weights='imagenet',  encoder_freeze=True)

input_shape = (None,None,1)
inputs = layers.Input(shape=input_shape,name='Custom_input1')

in1 = layers.Conv2D(3, 3,  padding="same")(inputs)
output = resnet_50(in1)
# output = layers.Conv2D(1, 3, activation="softmax", padding="same", dtype=tf.float64)(in2)
model = Model(inputs=inputs, outputs=output, name='Weaksuper_Res50')

# Compile and train

model.compile(optimizer='Adam',loss="binary_crossentropy", #sparse_ categorical,"binary_crossentropy"
        metrics=["accuracy"],
    )


config['start_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
logz= f"{config['base_path']}/{fname}/{config['start_time']}_logs/"

checkpoint_filepath = f"{config['base_path']}//Res50_pretrained_model//Res50_pretrained_model_Checkpoint{config['start_time']}.h5"

callbacks = [
   ModelCheckpoint(checkpoint_filepath, save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=0.000005, verbose= 1),
    EarlyStopping(monitor="val_loss", patience=30, verbose=1), 
    TensorBoard(log_dir = logz,histogram_freq = 1,profile_batch = '1,70', embeddings_freq=50),
    #WandbCallback()
]


history = model.fit(train_ds, epochs = config['epochs'],validation_data = val_ds, callbacks = callbacks)

model.load_weights(checkpoint_filepath)

_, accuracy, test_iou = model.evaluate(test_ds)
print(f"Test accuracy: {round(accuracy * 100, 2)}%")

model_save_path = f'{base_path}/Res50_pretrained_model/Res50_pretrained_model_{accuracy*100:.2f}_{config["start_time"]}.h5'
model.save(model_save_path)

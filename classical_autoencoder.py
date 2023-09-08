# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 15:52:53 2023

@author: i368o351
"""

# This is wrong for now but will be modified

from tensorflow.keras import layers
from tensorflow import keras
#from keras import backend as K
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm,colors

import os
import random
from scipy.io import loadmat
from scipy.ndimage import median_filter as sc_med_filt
# import tensorflow_probability as tfp
# from keras.metrics import MeanIoU
# from sklearn.metrics import roc_auc_score

import glob
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,ReduceLROnPlateau
from datetime import datetime


import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()

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
# tf.keras.mixed_precision.set_global_policy('mixed_float16')


model_name = 'auto_encoder'

use_wandb = True
time_stamp = '11_April_23_1415' #datetime.strftime( datetime.now(),'%d_%B_%y_%H%M') #'13_December_22_2204'

if use_wandb:    
    ## WandB config
    import wandb
    from wandb.keras import WandbCallback    
    
    wandb.init( project="my-test-project", entity="ibksolar", name= model_name + '_' + time_stamp,config ={})
    config = wandb.config
else:
    config ={}


# PATHS
# Path to data
base_path = r'U:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data'  # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'
# train_aug = os.path.join(base_path,'augmented_plus_train_data\*.mat')
# val_path = os.path.join(base_path,'val_data\*.mat')
# test_path = os.path.join(base_path,'test_data\*.mat')   

train_path = os.path.join(base_path,'larger_unsuper_data\good_ones\*.png')

# Create tf.data.Dataset
config['Run_Note'] = 'First run of VAE'
config['batch_size'] = 4

config['img_y'] = 1664 #1664 , 416
config['img_x'] = 256 #256, 64
config['dropout_rate'] = 0.2

# Training params
#config={}
config['img_channels'] = 1

config['num_classes'] = 1
config['epochs'] = 150
config['learning_rate'] = 1e-3
config['base_path'] = base_path
SEED = 42
AUTO = tf.data.experimental.AUTOTUNE

latent_dim = 64
num_embeddings = 128

# =============================================================================
## Function for creating dataloader
def read_mat(filepath):
    def _read_mat(filepath):
        
        filepath = bytes.decode(filepath.numpy())      
        mat_file = loadmat(filepath)
        echo = tf.cast(mat_file['echo_tmp']) #, dtype=tf.float64
        
        echo = tf.expand_dims(echo, axis=-1)
        
        if config['img_channels']> 1:            
            echo = tf.image.grayscale_to_rgb(echo)         
        
        shape0 = echo.shape        
        return echo,np.asarray(shape0)    
    output = tf.py_function(_read_mat,[filepath],[tf.float32,tf.int32]) #,tf.double
    shape = output[1]
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([config['img_y'],config['img_x'], config['img_channels'] ])    
  
    return data0

def _read_png(filepath):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_png(img)
    img = img/255
    return img,img

train_ds = tf.data.Dataset.list_files(train_path,shuffle=True) #'*.mat'
train_ds = train_ds.map(_read_png,num_parallel_calls=8)
train_ds = train_ds.batch(config['batch_size'],drop_remainder=True).prefetch(AUTO) #.shuffle(buffer_size = 100 * config['batch_size'])
# =============================================================================

input_shape = (config['img_y'], config['img_x'], config['img_channels'])
encoder_inputs = keras.Input(shape=input_shape ) #(28,28,1)
ResNet_50_model = sm.Unet(backbone_name='resnet50', encoder_weights='imagenet', encoder_freeze=True)    

in1 = layers.Conv2D(3, (3, 3), activation='relu', padding='same' )(encoder_inputs)

x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(in1)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoder = layers.MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (104, 16, 1) i.e. 104*16-dimensional

x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoder)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoder)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)

decoder = layers.Conv2D(1, (3, 3), activation='softmax', padding='same')(x)

model  = keras.Model(encoder_inputs,decoder) 

config['start_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M') 
logz= f"{config['base_path']}/{model_name}/{config['start_time']}_logs/"
callbacks = [
   ModelCheckpoint(f"{config['base_path']}//{model_name}//{model_name}_Checkpoint{time_stamp}.h5", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.25, patience=10, min_lr=0.00001, verbose= 1),
    EarlyStopping(monitor="val_loss", patience=30, verbose=1), 
    TensorBoard(log_dir = logz,histogram_freq = 1,profile_batch = '1,70', embeddings_freq=50),
    WandbCallback()
]

opt2 = tfa.optimizers.AdamW(learning_rate= config['learning_rate'], weight_decay=0.04)
model.compile(optimizer=opt2, loss ='mse')

history = model.fit(train_ds, epochs=100, batch_size=config['batch_size']) #, callbacks=callbacks


test_data = glob.glob(os.path.join(base_path,'test_data/*.mat'))

idx = np.random.choice(len(test_data), 10)

test_images = [np.expand_dims(np.expand_dims(loadmat(test_data[ix])['echo_tmp'],-1),0) for ix in idx]

reconstructions_test = model.predict(test_images[0])









































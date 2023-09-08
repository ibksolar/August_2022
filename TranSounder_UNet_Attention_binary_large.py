# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 15:06:56 2022

This is yet to work: Need to fix Squeeze attention module

@author: i368o351
"""

from tensorflow.keras import layers
from tensorflow import keras
from keras import backend as K

import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

import os
import random
from scipy.io import loadmat,savemat
from scipy.ndimage import median_filter as sc_med_filt

from keras.metrics import MeanIoU
from sklearn.metrics import roc_auc_score


import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()


import glob
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,ReduceLROnPlateau
#import segmentation_models as sm
from datetime import datetime
# from focal_loss import SparseCategoricalFocalLoss

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

time_stamp = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')


use_wandb = True
if use_wandb:
    ## WandB config
    import wandb
    from wandb.keras import WandbCallback
    wandb.init( project="my-test-project", entity="ibksolar", name='TransSounder_large'+time_stamp,config ={})
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
    

# PATHS
# Path to data
base_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\new_trainJuly'  # < == FIX HERE  Full_size_data e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'
train_aug = os.path.join(base_path,'augmented_data\*.mat')
train_path = os.path.join(base_path,'train_data\*.mat')
val_path = os.path.join(base_path,'val_data1\*.mat')
test_path = os.path.join(base_path,'test_data\*.mat')   

# Create tf.data.Dataset
config['batch_size'] = 2

config['img_y'] = 416 #1664
config['img_x'] = 64 #256

# Training params
#config={}
config['img_channels'] = 1

config['num_classes'] = 30
config['epochs'] = 500
config['learning_rate'] = 1e-3
config['base_path'] = base_path
SEED = 42
AUTO = tf.data.experimental.AUTOTUNE


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
    data1.set_shape([config['img_y'],config['img_x'],config['img_channels']])#,30,config['num_classes']    
    return data0,data1

# =============================================================================

# =============================================================================
## Function for test and validation dataset    
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
            layer = tf.keras.utils.to_categorical(layer, config['num_classes'] )
        
        shape0 = echo.shape        
        return echo,layer,np.asarray(shape0)     
    
    output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double, tf.int64])
    shape = output[2]
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([config['img_y'],config['img_x'], config['img_channels'] ])
    
    data1 = output[1]   
    data1.set_shape([config['img_y'],config['img_x'],config['num_classes']]) #,30, ,config['num_classes']    
    return data0,data1

train_ds = tf.data.Dataset.list_files(train_aug,shuffle=True) #'*.mat'
train_ds = train_ds.map(read_mat,num_parallel_calls=8)
train_ds = train_ds.batch(config['batch_size'],drop_remainder=True).prefetch(AUTO) #.shuffle(buffer_size = 100 * config['batch_size'])

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


## Transformer functions and hyper param
config['dropout_rate'] = 0.1
config['num_heads'] = 12

num_transformer_layers = 8
hidden_units = [512,256]

num_patches = config['img_y']
embed_dim = config['img_x']


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches = config['img_y'], embed_dim = config['img_x'], **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches        
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=embed_dim
        )

    def call(self, patch):        
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        positions = self.position_embedding(positions)
        # encoded = patch + tf.transpose(positions,perm= [0,1])
        encoded = patch + positions
        return encoded
    
    # Override function to avoid error while saving model
    def get_config(self):
        config = super().get_config().copy()
        config.update(
        {               
            "num_patches": num_patches,
            "embed_dim": embed_dim,
        }
    )
        return config

def mlp(x, hidden_units, dropout_rate = config['dropout_rate']):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.leaky_relu,)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# def squeeze_excitation(feature_map):
#     f_shape = (feature_map).shape
    
#     f_map = layers.GlobalAveragePooling2D() (feature_map)
#     f_map = tf.keras.layers.Reshape((1, 1, f_shape[-1])) (f_map) 
#     f_map = layers.Dense(f_shape[-1]//4, use_bias = False, activation='relu')(f_map)
#     f_map = layers.Dense(f_shape[-1],activation='sigmoid', use_bias = False)(f_map)     
    
#     feature_map = feature_map * f_map 
    
    # return feature_map

def BiFusion_module(attn_data, conv_data):
    out_shape = (attn_data).shape[-1]
    
    # sq_attn = squeeze_excitation(attn_data)
    # sq_conv = squeeze_excitation(conv_data)
    
    conv_both = attn_data + conv_data
    conv_both = layers.Conv2D(out_shape, 3, padding='same')(conv_both)
    
    return  conv_both #sq_attn + sq_conv +





#Build the model  
input_shape = (config['img_y'], config['img_x'],config['img_channels'])
      
inputs = tf.keras.layers.Input(shape = input_shape)
#s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

# Do attention on raw input
attn_large  = tf.keras.layers.Conv2D(1, (config['img_y'], 1), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
attn_large2 = tf.squeeze(attn_large, axis = -1)

encoded = PatchEncoder(config['img_x'],config['img_y'] ) (attn_large) # Check this again
encoded_patches = tf.reduce_mean(encoded, axis = -1)

# Create multiple layers of the Transformer block.
for _ in range(num_transformer_layers):
    
    final_shape = [encoded_patches.shape[-1]]
    # Layer normalization 1.
    x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    
    # Create a multi-head attention layer.
    attention_output = layers.MultiHeadAttention(
        num_heads=config['num_heads'], key_dim=config['img_x'], dropout=0.1
    )(x1, x1)    
    
    # Skip connection 1.
    x2 = layers.Add()([attention_output, encoded_patches])
    
    # Layer normalization 2.
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2) 
    
    # MLP
    mlp_out = mlp(x3,hidden_units+final_shape)
    
    # Skip connection 3.
    encoded_patches = layers.Add()([x3, mlp_out])

# positions = tf.range(start=0, limit=config['img_x'],  delta=1)
# position_embedding = layers.Embedding(input_dim= config['img_x'], output_dim=config['img_y']) (positions)
# position_embedding1 = position_embedding[None,:,:]
# position_embedding = tf.transpose( position_embedding[None,:,:]) ,perm=[0,2,1])

# Use pre-trained model
base_model = sm.Unet(backbone_name='resnet50', encoder_weights='imagenet', encoder_freeze=True)
c1 = tf.keras.layers.Conv2D(3, (17, 13), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
c1_out = base_model(c1)

c1 = tf.keras.layers.BatchNormalization()(c1_out)

#Contraction path
c1 = tf.keras.layers.Conv2D(16, (17, 13), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.BatchNormalization()(c1)  
c1 = tf.keras.layers.Activation('relu')(c1)

# c1 = tf.keras.layers.Conv2D(16, (17, 13), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
# c1 = tf.keras.layers.Dropout(0.1)(c1)
# c1 = tf.keras.layers.BatchNormalization()(c1)
# c1 = tf.keras.layers.Activation('relu')(c1)

c1 = tf.keras.layers.Conv2D(16, (7, 5), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.BatchNormalization()(c1)
c1 = tf.keras.layers.Activation('relu')(c1)

c1 = tf.keras.layers.Conv2D(16, (7, 5), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.BatchNormalization()(c1)
c1 = tf.keras.layers.Activation('relu')(c1)

c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.BatchNormalization()(c2)
c2 = tf.keras.layers.Activation('relu')(c2)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.BatchNormalization()(c3)
c3 = tf.keras.layers.Activation('relu')(c3)

c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.BatchNormalization()(c4)
c4 = tf.keras.layers.Activation('relu')(c4)

c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.BatchNormalization()(c5)
c5 = tf.keras.layers.Activation('relu')(c5)

c5 = tf.keras.layers.Conv2D(config['img_y']//16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)


# Using Convolution for patching
attn_5  = tf.keras.layers.Conv2D(config['img_y']//16, (config['img_y']//16, 1), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
attn_5 = PatchEncoder(config['img_x']//16,config['img_y']//16 ) (attn_5)

BF1 = BiFusion_module(attn_5, c5) #, config['img_y']//16

#Expansive path 
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.BatchNormalization()(c6)
c6 = tf.keras.layers.Activation('relu')(c6)

c6 = tf.keras.layers.Conv2D(config['img_y']//8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
attn_6 = layers.Conv2DTranspose(config['img_y']//8, 3, strides = (2,2), padding='same')(BF1)
BF2 = BiFusion_module(attn_6, c6) #, config['img_y']//8
 
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.BatchNormalization()(c7)
c7 = tf.keras.layers.Activation('relu')(c7)

c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

c7 = tf.keras.layers.Conv2D(config['img_y']//4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
attn_7 = layers.Conv2DTranspose(config['img_y']//4, 3, strides = (2,2), padding='same')(BF2)
BF3 = BiFusion_module(attn_7, c7) #, config['img_y']//4
 
u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.BatchNormalization()(c8)
c8 = tf.keras.layers.Activation('relu')(c8)

c8 = tf.keras.layers.Conv2D(config['img_y']//2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
attn_8 = layers.Conv2DTranspose(config['img_y']//2, 3, strides = (2,2), padding='same')(BF3)
BF4 = BiFusion_module(attn_8, c8) #, config['img_y']//2
 
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.4)(c9)
c9 = tf.keras.layers.BatchNormalization()(c9)
c9 = tf.keras.layers.Activation('relu')(c9)

c9 = tf.keras.layers.Conv2D(config['img_y'], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
attn_9 = layers.Conv2DTranspose(config['img_y'], 3, strides = (2,2), padding='same')(BF4)
BF5 = BiFusion_module(attn_9, c9) #, config['img_y']

encoded_patches_final = tf.expand_dims(encoded_patches, axis=-1)
encoded_patches_final = tf.keras.layers.Conv2D(config['img_y'], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(encoded_patches_final)

combined = c9 + attn_9 + BF5 + encoded_patches_final

final_activation = "sigmoid" if config['num_classes'] == 1 else "softmax"


output_comb = tf.keras.layers.Conv2D(config['num_classes'], (1, 1), padding="same", dtype=tf.float32, activation=final_activation )(combined)
 
output_conv = tf.keras.layers.Conv2D(config['num_classes'], (1, 1), padding="same", dtype=tf.float32, activation=final_activation )(c9) #sigmoid, , activity_regularizer='l2', activation='softmax'
output_BF = tf.keras.layers.Conv2D(config['num_classes'], (1, 1), padding="same", dtype=tf.float32, activation= final_activation)(BF5)
output_resnet = tf.keras.layers.Conv2D(config['num_classes'], (1, 1), padding="same", dtype=tf.float32, activation = final_activation )(c1_out)



output_attn = tf.keras.layers.Conv2D(config['num_classes'], (1, 1), padding="same", dtype=tf.float32, activation= final_activation )(encoded_patches_final)

model = tf.keras.Model(inputs,[output_comb, output_conv,output_BF,output_resnet,output_attn])
opt = keras.optimizers.Adam(learning_rate=config['learning_rate'])
opt2 = tfa.optimizers.AdamW(weight_decay = 0.001, learning_rate = config['learning_rate'])


loss = "binary_crossentropy" if config['num_classes']==1 else "categorical_crossentropy"

model.compile(optimizer= opt2, loss= loss, loss_weights=[0.3,0.15,0.15,0.15,0.15], metrics=['accuracy',sm.metrics.iou_score]) # sparse_categorical_crossentropy,jaccard_distance,binary_crossentropy,tf.keras.losses.KLDivergence(),

config['base_path'] = base_path
config['start_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
logz= f"{config['base_path']}/TransSounder_large//{config['start_time']}_logs/"
callbacks = [
   ModelCheckpoint(f"{config['base_path']}//TransSounder_large//TransSounder_Checkpoint{time_stamp}.h5", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=0.000005, verbose= 1),
    EarlyStopping(monitor="val_loss", patience=30, verbose=1), 
    TensorBoard(log_dir = logz,histogram_freq = 1,profile_batch = '1,70', embeddings_freq=50),
    WandbCallback()
]


config['start_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
print(f"Training start time {config['start_time']} " )

model.fit(train_ds, epochs=config['epochs'], validation_data=val_ds, callbacks=callbacks)

config['end_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
print(f"Training end time {config['end_time']} " )

_,acc = model.evaluate(test_ds)
model.save(f"{config['base_path']}//TransSounder_large//TransSounder_acc_{acc:.2f}_GOOD_{time_stamp}.h5")


model_val_data_path = os.path.join(base_path,'test_data\*.mat')
model_val_data = glob.glob(model_val_data_path)


## Final prediction correcting code
from itertools import groupby

def fix_final_prediction(a, a_final):
    for col_idx in range(a_final.shape[1]):
        repeat_tuple = [ (k,sum(1 for _ in groups)) for k,groups in groupby(a_final[:,col_idx]) ]
        rep_locs = np.cumsum([ item[1] for item in repeat_tuple])
        
        # Temporary hack
        rep_locs[-1] = rep_locs[-1] - 1
        
        locs_to_fix = [ (elem[1],rep_locs[idx]) for idx,elem in enumerate(repeat_tuple) if elem[0]== 1 and elem[1]>1 ]
        
        for elem0 in locs_to_fix:
            check_idx = list(range(elem0[1]-elem0[0],elem0[1]+1))
            max_loc = check_idx[0] + np.argmax(a[elem0[1]-elem0[0]:elem0[1], col_idx])
            check_idx.remove(max_loc)            
            a_final[check_idx,col_idx] = 0
    
    return a_final

### Create vec_layer function
def create_vec_layer(raster):
    vec_layer = []
    for iter in range(raster.shape[1]):
        temp = np.nonzero(raster[:,iter])
        vec_layer.append(temp[0])
        
    return np.array(list(itertools.zip_longest(*vec_layer, fillvalue=0)))
    

# # Visualize result of model prediction for "unseen" echogram during training

batch_idx = random.randint(1,len(model_val_data)) # Pick any of the default batch

for idx in range(1,10):
  predict_data = loadmat(model_val_data[batch_idx+idx])
  a01,a_gt0 = predict_data['echo_tmp'], predict_data['raster']
  if config['img_channels']  >1:
      a0 = np.stack((a01,)*3,axis=-1)
      res0 = model.predict ( np.expand_dims(a0,axis=0))
  else:
      res0 = model.predict ( np.expand_dims(np.expand_dims(a0,axis=0),axis=3) ) 
      
  res0 = res0.squeeze()
  #res0_final = np.argmax(res0,axis=2)
  res0_final = np.where(res0>0.05,1,0)
  
  res0_final1 = np.arange(1,config['img_y']+1).reshape(config['img_y'],1) * fix_final_prediction(res0,res0_final)

  f, axarr = plt.subplots(1,3,figsize=(20,20))

  axarr[0].imshow(a01.squeeze(),cmap='gray_r')
  axarr[0].set_title( f'Echo {os.path.basename(model_val_data[batch_idx])}') #.set_text
  
  axarr[1].imshow(res0_final1.astype(bool), cmap='gray_r' )
  axarr[1].set_title('Prediction')
  
  axarr[2].imshow(a_gt0.squeeze().astype(bool),cmap='gray_r') # gt
  axarr[2].set_title( f'Ground truth {os.path.basename(model_val_data[batch_idx])}') #.set_text







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
                  
            
layer_thickness = compute_layer_thickness(model, test_ds)        
        
        
(np.mean(layer_thickness,axis =1)).round()























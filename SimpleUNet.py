# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 15:06:56 2022

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
from scipy.io import loadmat
from scipy.ndimage import median_filter as sc_med_filt

from keras.metrics import MeanIoU
from sklearn.metrics import roc_auc_score

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


## WandB config
import wandb
from wandb.keras import WandbCallback
time_stamp = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')


use_wandb = True
if use_wandb:
    wandb.init( project="my-test-project", entity="ibksolar", name='EchoViT2'+time_stamp,config ={})
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
base_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\new_trainJuly'  # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'
train_path = os.path.join(base_path,'train_data\*.mat')
val_path = os.path.join(base_path,'val_data1\*.mat')
test_path = os.path.join(base_path,'test_data\*.mat')   

# Create tf.data.Dataset
config['batch_size'] = 8

# Training params
#config={}
config['img_channels'] = 3

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
        
        layer = tf.cast(mat_file['semantic_seg'], dtype=tf.float64)
        
        # Data Augmentation
        
        if tf.random.uniform(())> 0.3:
            aug_type = tf.random.uniform((1,1),minval=1, maxval=8,dtype=tf.int64).numpy()
            
            if aug_type == 1:
                echo = tf.experimental.numpy.fliplr(echo)
                layer = tf.experimental.numpy.fliplr(layer)
            
            elif aug_type == 2: # Constant offset
                echo = echo - 0.3
            
            elif aug_type == 3: # Random noise
                echo = echo - tf.random.normal(shape=(416,64,config['img_channels']),stddev=0.5,dtype=tf.float64)
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
                   
        
        layer = tf.keras.utils.to_categorical(layer, config['num_classes'] )
        shape0 = echo.shape #mat_file['echo_tmp'].shape
        
        return echo,layer,np.asarray(shape0)
    output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double, tf.int64])
    shape = output[2]
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([416,64,config['img_channels']])
    
    data1 = output[1]   
    data1.set_shape([416,64,30])#,30    
    return data0,data1

# =============================================================================
## Function for test and validation dataset    
def read_mat(filepath):
    def _read_mat(filepath):
        
        filepath = bytes.decode(filepath.numpy())      
        mat_file = loadmat(filepath)
        echo = tf.cast(mat_file['echo_tmp'], dtype=tf.float64)
        layer = tf.cast(mat_file['semantic_seg'], dtype=tf.float64)      

        layer = tf.keras.utils.to_categorical(layer, config['num_classes'] )
        shape0 = mat_file['echo_tmp'].shape        
        return echo,layer,np.asarray(shape0)     
    
    output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double, tf.int64])
    shape = output[2]
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([416,64])
    
    data1 = output[1]   
    data1.set_shape([416,64,30 ]) #,30   
    return data0,data1

train_ds = tf.data.Dataset.list_files(train_path,shuffle=True) #'*.mat'
train_ds = train_ds.map(read_mat_train,num_parallel_calls=8)
train_ds = train_ds.batch(config['batch_size'],drop_remainder=True).prefetch(AUTO) #.shuffle(buffer_size = 100 * config['batch_size'])

val_ds = tf.data.Dataset.list_files(val_path,shuffle=True)
val_ds = val_ds.map(read_mat_train,num_parallel_calls=8)
val_ds = val_ds.batch(config['batch_size'],drop_remainder=True).cache().prefetch(AUTO)

test_ds = tf.data.Dataset.list_files(test_path,shuffle=True)
test_ds = test_ds.map(read_mat_train,num_parallel_calls=8)
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



#Build the model  
input_shape = (416, 64,config['img_channels'])
      
inputs = tf.keras.layers.Input(shape = input_shape)
#s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)


#Contraction path
c1 = tf.keras.layers.Conv2D(16, (17, 13), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.BatchNormalization()(c1)  
c1 = tf.keras.layers.Activation('relu')(c1)

c1 = tf.keras.layers.Conv2D(16, (17, 13), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.BatchNormalization()(c1)
c1 = tf.keras.layers.Activation('relu')(c1)

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

c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path 
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.BatchNormalization()(c6)
c6 = tf.keras.layers.Activation('relu')(c6)

c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.BatchNormalization()(c7)
c7 = tf.keras.layers.Activation('relu')(c7)

c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.BatchNormalization()(c8)
c8 = tf.keras.layers.Activation('relu')(c8)

c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.BatchNormalization()(c9)
c9 = tf.keras.layers.Activation('relu')(c9)

c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
 
outputs = tf.keras.layers.Conv2D(config['num_classes'], (1, 1), padding="same", dtype=tf.float32, activation='softmax' )(c9) #sigmoid, , activation='softmax'

model = tf.keras.Model(inputs,outputs)
opt = keras.optimizers.Adam(learning_rate=config['learning_rate'])
opt2 = tfa.optimizers.AdamW(weight_decay = 0.001, learning_rate = config['learning_rate'])

model.compile(optimizer= opt2, loss= tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy']) # sparse_categorical_crossentropy,jaccard_distance,binary_crossentropy,tf.keras.losses.KLDivergence(),

config['base_path'] = base_path
config['start_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
logz= f"{config['base_path']}/SimpleUNet//{config['start_time']}_logs/"
callbacks = [
   ModelCheckpoint(f"{config['base_path']}//SimpleUNet//SimpleUNet_Checkpoint{time_stamp}.h5", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=0.000005, verbose= 1),
    EarlyStopping(monitor="val_loss", patience=30, verbose=1), 
    TensorBoard(log_dir = logz,histogram_freq = 1,profile_batch = '1,70', embeddings_freq=50),
    WandbCallback()
]


model.fit(train_ds, epochs=config['epochs'], validation_data=val_ds, callbacks=callbacks)


_,acc = model.evaluate(test_ds)
model.save(f"{config['base_path']}//SimpleUNet//SimpleUNet_acc_{acc:.2f}_dice_coeff_{time_stamp}.h5")


model_val_data_path = os.path.join(base_path,'Test_data\*.mat')
model_val_data = glob.glob(model_val_data_path)



# # Visualize result of model prediction for "unseen" echogram during training

batch_idx = random.randint(1,len(model_val_data)) # Pick any of the default batch

for idx in range(1,10):
  predict_data = loadmat(model_val_data[batch_idx+idx])
  a0,a_gt0 = predict_data['echo_tmp'], predict_data['semantic_seg']
  if config['img_channels']  >1:
      a0 = np.stack((a0,)*3,axis=-1)
      res0 = model.predict ( np.expand_dims(a0,axis=0))
  else:
      res0 = model.predict ( np.expand_dims(np.expand_dims(a0,axis=0),axis=3) ) 
      
  res0 = res0.squeeze()
  res0_final = np.argmax(res0,axis=2)


  f, axarr = plt.subplots(1,3,figsize=(20,20))

  axarr[0].imshow(a0.squeeze(),cmap='gray_r')
  axarr[0].set_title( f'Echo {os.path.basename(model_val_data[batch_idx])}') #.set_text

  axarr[1].imshow(a_gt0.squeeze(),cmap='viridis') # gt
  axarr[1].set_title( f'Ground truth {os.path.basename(model_val_data[batch_idx])}') #.set_text

  axarr[2].imshow(res0_final, cmap='viridis' )
  axarr[2].set_title('Prediction')


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























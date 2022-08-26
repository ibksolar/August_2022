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
from focal_loss import SparseCategoricalFocalLoss
import glob

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
time_stamp = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')

if use_wandb:    
    ## WandB config
    import wandb
    from wandb.keras import WandbCallback    
    
    wandb.init( project="my-test-project", entity="ibksolar", name='ConvMixer_large'+time_stamp,config ={})
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

# PATHS
# Path to data
base_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data'  # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'
train_path = os.path.join(base_path,'train_data\*.mat')
val_path = os.path.join(base_path,'val_data\*.mat')
test_path = os.path.join(base_path,'test_data\*.mat')   

# Hyperparameters
config['Run_Note'] = 'ConvMixer_Segmentation_large'
config['batch_size'] = 4


# Training params
config['img_y'] = 416*4
config['img_x'] = 64*4

config['img_channels'] = 3
config['weight_decay'] = 0.0001

config['num_classes'] = 30
config['epochs'] = 500
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
                
              
        
        # Random hue
        # echo = tf.image.random_hue(echo,0.2, 123)
        #echo = tf.clip_by_value(echo, 0, 1)           
        
        # Zero random columns
        # col_idx,_ = tf.unique(tf.reshape(tf.random.uniform((1,10),minval=1, maxval=64,dtype=tf.int64),[10]) )
        # echo[:,col_idx] = tf.Variable(0, dtype=tf.float64)
        
        # # Zero random rows
        # row_idx,_ = tf.unique(tf.reshape(tf.random.uniform((1,20),minval=1, maxval=config['img_y'],dtype=tf.int64),[10]) )
        # echo[row_idx,:] = tf.Variable(0, dtype=tf.float64)
        

        #echo = tf.experimental.numpy.flipud(echo)
        #layer = tf.experimental.numpy.flipud(layer)                            
                   
        
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
        
        layer = tf.cast(mat_file['semantic_seg'], dtype=tf.float64)
        #layer = tf.cast( tf.cast(mat_file['raster'], dtype=tf.bool), dtype=tf.float64)
        
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
    data0.set_shape([1664,256,config['img_channels']])
    
    data1 = output[1]   
    data1.set_shape([1664,256,1 ])#,30,config['num_classes']    
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
        
        layer = tf.cast(mat_file['semantic_seg'], dtype=tf.float64)      
        
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



input_shape = (config['img_y'], config['img_x'], config['img_channels'])


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
     filters=64, depth=8, kernel_size=(17,15), patch_size=(1,config['img_y']), num_classes=config['num_classes']): #depth=8, kernel_size=5  patch_size=2,
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

    # Classification block.
    x = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(x)
    # x = layers.GlobalAvgPool2D()(x)
    x= layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal',strides=(2, 2), padding='same')(x)
    outputs = layers.Conv2D(config['num_classes'],(1,1), padding = 'same', dtype=tf.float64)(x) #softmax, sigmoid, activation="softmax", 

    return Model(inputs, outputs)



#========================================================================
# =============== MODEL TRAINING AND EVALUATION =========================
#========================================================================

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 

def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=config['learning_rate'], weight_decay=config['weight_decay']
    )

    model.compile(
        optimizer=optimizer,
        loss=loss, #sparse_ categorical,"binary_crossentropy"
        metrics=["accuracy"],
    )

    checkpoint_filepath = os.path.abspath(base_path+"/ConvMixer/checkpoint")
    checkpoint_callback = [
        ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.25, patience=10, min_lr=0.00005, verbose= 1),
        EarlyStopping(monitor="val_loss", patience=30, verbose=1)  ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config['epochs'],
        callbacks=[checkpoint_callback,], #WandbCallback()
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy = model.evaluate(test_ds)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, model


conv_mixer_model = get_conv_mixer_256_8()
history, conv_mixer_model = run_experiment(conv_mixer_model)


time_stamp = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')

_, accuracy = conv_mixer_model.evaluate(test_ds)

model_save_path = f'{base_path}/ConvMixer/SegAcc_{accuracy*100: .2f}_{time_stamp}.h5'

conv_mixer_model.save(model_save_path)



## Visualize result of model prediction for "unseen" echogram during training
model_val_data_path = os.path.join(base_path,'test_data\*.mat')
#model_val_data_path = os.path.join(base_path,'new_test\image\*.mat')
model_val_data = glob.glob(model_val_data_path)

batch_idx = random.randint(1,len(model_val_data)-10) # Pick any of the default batch
model = conv_mixer_model
for idx in range(10):
  predict_data = loadmat(model_val_data[batch_idx+idx])
  a01,a_gt0 = predict_data['echo_tmp'], predict_data['semantic_seg']
  
  if config['img_channels'] > 1:
    a0 = np.stack((a01,)*3,axis=-1)
    res0 = model.predict ( np.expand_dims(a0,axis=0))
  else:
    res0 = model.predict ( np.expand_dims(np.expand_dims(a01,axis=0),axis=3) )   
  # a0 = a[idx]
  # a_gt0 = a_gt[idx]
  # ( a0.shape, a_gt0.shape )

  
  res0 = res0.squeeze()
  res0_final = np.argmax(res0,axis=2)
  pred0_final = sc_med_filt( sc_med_filt(res0_final,size=7).T, size=7, mode='nearest').T


  f, axarr = plt.subplots(1,4,figsize=(20,20))

  axarr[0].imshow(a01.squeeze(),cmap='gray_r')
  axarr[0].set_title( f'Echo {os.path.basename(model_val_data[batch_idx+idx])}') #.set_text

  axarr[1].imshow(a_gt0.squeeze(),cmap='viridis') # gt
  axarr[1].set_title( 'Ground truth') #.set_text

  axarr[2].imshow(res0_final, cmap='viridis') 
  axarr[2].set_title('Prediction')

  axarr[3].imshow(pred0_final, cmap='viridis') 
  axarr[3].set_title('Filtered Prediction')








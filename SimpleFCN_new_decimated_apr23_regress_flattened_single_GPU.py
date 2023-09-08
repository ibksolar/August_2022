# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 09:53:50 2022
Simple FCN
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

# from keras.metrics import MeanIoU
# from sklearn.metrics import roc_auc_score

import glob
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,ReduceLROnPlateau
#import segmentation_models as sm
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
tf.keras.mixed_precision.set_global_policy('mixed_float16')

model_name = 'SimpleFCNet_NewDecimated_Apr23_regress_2D_SingleGPU'

time_stamp = '11_May_23_1409' #datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
use_wandb = True
if use_wandb:    
    ## WandB config
    import wandb
    from wandb.keras import WandbCallback    
    
    wandb.init( project="my-test-project", entity="ibksolar", name= model_name + time_stamp,config ={})
    config = wandb.config


# PATHS
# Path to data
# base_path = r'U:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data'  # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'

base_path = r'V:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\SR_Dataset_v1\Dec'
train_path = os.path.join(base_path,'train_data\*.mat')
# train_aug_path = os.path.join(base_path,'augmented_plus_train_data\*.mat')
val_path = os.path.join(base_path,'val_data\*.mat')
test_path = os.path.join(base_path,'test_data\*.mat')   

# Create tf.data.Dataset
config['batch_size'] = 8
config['num_classes'] = 1
config['num_layers'] = 32


# Training params
config['img_y'] = 416 # Decimated in fast time
config['img_x'] = 64*4

config['img_channels'] = 1
config['weight_decay'] = 0.0001


config['epochs'] = 20
config['learning_rate'] = 1e-3
config['base_path'] = base_path
SEED = 42
AUTO = tf.data.experimental.AUTOTUNE




# =============================================================================
def regress_read_mat(filepath):
    def _read_mat(filepath):
        
        dtype = tf.float64
        
        filepath = bytes.decode(filepath.numpy())      
        mat_file = loadmat(filepath)
        echo = tf.cast(mat_file['echo_tmp'], dtype= dtype) #, dtype=tf.float64
        
        echo = tf.expand_dims(echo, axis=-1)
        
        layer = tf.cast( tf.cast(mat_file['new_raster'], dtype=tf.bool), dtype = dtype)
        layer = tf.expand_dims(layer, axis=-1)
        
        layer3D = tf.cast(mat_file['regress_GT'], dtype = dtype) 
        layer3D = tf.reshape(layer3D, shape = (1,1,-1) )
        # layer3D = tf.expand_dims(layer3D, axis=-1)
        
        layer_diff = tf.cast(100*mat_file['layer_diff'], dtype = dtype) 
        layer_diff = tf.reshape(layer_diff, shape = (1,1,-1) )
        # layer_diff = tf.expand_dims(layer_diff, axis=-1)
        
        shape0 = echo.shape        
        return echo,layer,layer3D,layer_diff, np.asarray(shape0)     
    
    output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double,tf.double,tf.double, tf.int64]) #,tf.double, tf.half
    shape = output[4]
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([config['img_y'],config['img_x'], config['img_channels'] ])
    
    data1 = output[1]  
    data1.set_shape([config['img_y'],config['img_x'], config['img_channels'] ])
    
    data2 = output[2] 
    # data2.set_shape([config['img_y']//13,config['img_x'], 1]) #,30, ,config['num_classes']  
    data2.set_shape([1,1, config['num_layers'] * config['img_x'] ])
    
    data3 = output[3] 
    # data3.set_shape([config['img_y']//13,config['img_x'], 1])
    data3.set_shape([1,1, config['num_layers'] * config['img_x'] ])
    
    return data0,{'raster':data1, 'raster3D':data2, 'layer_diff':data3}




#=============================================================================
## Function for creating dataloader   
def read_mat(filepath):
    def _read_mat(filepath):
        
        filepath = bytes.decode(filepath.numpy())      
        mat_file = loadmat(filepath)
        echo = tf.cast(mat_file['echo_tmp'], dtype=tf.float64) #, dtype=tf.float64
        
        echo = tf.expand_dims(echo, axis=-1)        
        if config['img_channels'] > 1:
            echo = tf.image.grayscale_to_rgb(echo)
        
        # layer = tf.cast(mat_file['raster'], dtype=tf.float64)      
        layer = tf.cast(mat_file['new_raster'])
        
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
train_ds = train_ds.map(regress_read_mat,num_parallel_calls=8)
train_ds = train_ds.batch(config['batch_size'],drop_remainder=True).prefetch(AUTO) #.shuffle(buffer_size = 100 * config['batch_size'])

val_ds = tf.data.Dataset.list_files(val_path,shuffle=True)
val_ds = val_ds.map(regress_read_mat,num_parallel_calls=8)
val_ds = val_ds.batch(config['batch_size'],drop_remainder=True).cache().prefetch(AUTO)

test_ds = tf.data.Dataset.list_files(test_path,shuffle=True)
test_ds = test_ds.map(regress_read_mat,num_parallel_calls=8)
test_ds = test_ds.batch(config['batch_size'],drop_remainder=True).cache().prefetch(AUTO)

# # train_shape = [ ( tf.shape(item[0]).numpy(),tf.shape(item[1]).numpy() ) for item in train_ds.take(1) ]
# # train_shape = train_shape[0]

# print(f' X_train train shape {train_shape[0]}')
# print(f' Training target shape {train_shape[1]}')



# Custom Loss
newFocalLoss = sm.losses.BinaryFocalLoss() #SparseCategoricalFocalLoss(gamma=2, from_logits= True)





#Build the model 
# input_shape = (config['img_y'], config['img_x'], config['img_channels'])

input_shape = (None, None, config['img_channels'])

ResNet_50_model = sm.Unet(backbone_name='resnet50', encoder_weights='imagenet', encoder_freeze=True, input_shape = (None,None,3) ) #   
inputs = keras.Input(shape=input_shape ) #(28,28,1)
in1 = layers.Conv2D(3, (3, 3), activation='relu', padding='same' )(inputs)       
in2 = ResNet_50_model(in1)

#s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
#inputs = tf.expand_dims(inputs, axis=-1)

#Contraction path
c1 = tf.keras.layers.Conv2D(16, (17, 13), activation='relu', kernel_initializer='he_normal', padding='same')(in2)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.BatchNormalization()(c1)  
c1 = tf.keras.layers.Activation('relu')(c1)

c1 = tf.keras.layers.Conv2D(32, (17, 13), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.BatchNormalization()(c1)
c1 = tf.keras.layers.Activation('relu')(c1)

c1 = tf.keras.layers.Conv2D(32, (7, 5), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.BatchNormalization()(c1)
c1 = tf.keras.layers.Activation('relu')(c1)

c1 = tf.keras.layers.Conv2D(64, (7, 5), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.BatchNormalization()(c1)
c1 = tf.keras.layers.Activation('relu')(c1)

c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
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
# u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
# c6 = tf.keras.layers.BatchNormalization()(c6)
c6 = tf.keras.layers.Activation('relu')(c6)

c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
# u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
# c7 = tf.keras.layers.BatchNormalization(epsilon=1e-3)(c7)
c7 = tf.keras.layers.Activation('relu')(c7)

c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
# u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
# c8 = tf.keras.layers.BatchNormalization(epsilon=1e-3)(c8)
c8 = tf.keras.layers.Activation('relu')(c8)

c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c8)
# u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
# c9 = tf.keras.layers.BatchNormalization(epsilon=1e-3)(c9)
c9 = tf.keras.layers.Activation('relu')(c9)

x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
skip_x_c5 = tf.keras.layers.Conv2DTranspose(16,3,16)(c5) # take a copy of bottleneck and upsample
x = x + skip_x_c5
x = tf.keras.layers.BatchNormalization(epsilon=1e-3)(x)
x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
 
# outputs = tf.keras.layers.Conv2D(config['num_classes'], (1, 1), padding="same", dtype= tf.float32, activation='sigmoid' )(c9) #sigmoid , activation='softmax'
bin_output = layers.Conv2D(config['num_classes'],(1,1), padding = 'same',activation="sigmoid", dtype=tf.float64 , name= "raster" )(x) 

# Regression output
regress_init = layers.Conv2D(256, (17,15), activation='relu', padding="same") (x)
regress_init = layers.Conv2D(128, 5, activation='relu', padding="same") (regress_init) 
 
cols_out = layers.Conv2D(64, (config['img_y'],1), activation='relu', padding="same") (regress_init)
rows_out = layers.Conv2D(64, (1,config['img_x']), activation='relu', padding="same") (regress_init)   

regress_out1 = cols_out + rows_out
regress_out1 = layers.BatchNormalization(epsilon=1e-3)(regress_out1)

regress_out1 = layers.MaxPool2D(pool_size=(13,1)) (regress_out1)
regress_out = layers.Conv2D( 16, (1,1), padding="same", activation='relu',)(regress_out1)
regress_out = layers.Conv2D( 1, (1,1), padding="same",  activation='relu' )(regress_out) # activation='sigmoid', name= "raster3D", 
# regress_out = layers.BatchNormalization(epsilon=1e-3)(regress_out)
# regress_out = layers.Dropout(0.3) (regress_out)

regress_out = layers.GlobalAveragePooling2D()(regress_out)
regress_out = layers.Dense(config['num_layers'] * config['img_x'], name= "raster3D", activation="linear", dtype = tf.float64)(regress_out)


regress_out2 = layers.Conv2D( 16, (1,1), padding="same", activation='relu',)(regress_out1)
layer_diff_out = layers.Conv2D( 1, (1,1), padding="same", dtype = tf.float64, activation='relu', )(regress_out2)
# layer_diff_out = layers.BatchNormalization(epsilon=1e-3)(layer_diff_out)
# layer_diff_out = layers.Dropout(0.3) (layer_diff_out) 

layer_diff_out = layers.GlobalAveragePooling2D()(layer_diff_out)
layer_diff_out = layers.Dense(config['num_layers'] * config['img_x'], name= "layer_diff", activation="linear", dtype = tf.float64)(layer_diff_out)




model = tf.keras.Model(inputs,[bin_output, regress_out, layer_diff_out] )
opt = tfa.optimizers.AdamW(learning_rate=config['learning_rate'], weight_decay = config['weight_decay'])
model.compile(optimizer= opt, loss=  ["binary_crossentropy", "mean_squared_error", "mean_squared_error"], metrics=['accuracy']) #  loss_weights=[1,10,10], mean_squared_error, sparse_categorical_crossentropy,jaccard_distance,binary_crossentropy,tf.keras.losses.KLDivergence(),

config['base_path'] = base_path
config['start_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
logz= f"{config['base_path']}/{model_name}//{config['start_time']}_logs/" #f"{config['base_path']}/SimpleUNet//{config['start_time']}_logs/"
callbacks = [
   ModelCheckpoint(f"{config['base_path']}//{model_name}//{model_name}_Checkpoint{time_stamp}.h5", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.25, patience=10, min_lr=0.000005, verbose= 1),
    EarlyStopping(monitor="val_loss", patience=30, verbose=1), 
    TensorBoard(log_dir = logz,histogram_freq = 1,profile_batch = '1,70', embeddings_freq=50),
    WandbCallback()
]


history = model.fit(train_ds, epochs=config['epochs'], validation_data=val_ds, callbacks=callbacks)

all_acc = model.evaluate(test_ds)
acc = all_acc[3] # Need to confirm this
model.save(f"{config['base_path']}//{model_name}//{model_name}_{acc:.2f}_{time_stamp}.h5")

# Custom colormap
custom_cm = cm.Blues(np.linspace(0,1,30))
custom_cm = colors.ListedColormap(custom_cm[10:,:-1])

## Visualize result of model prediction for "unseen" echogram during training
model_val_data_path = os.path.join(base_path,'test_data\*.mat')
model_val_data = glob.glob(model_val_data_path)

batch_idx = random.randint(1,len(model_val_data)) # Pick any of the default batch

for idx in range(1,10):
  predict_data = loadmat(model_val_data[batch_idx+idx])
  a0,a_gt0 = predict_data['echo_tmp'], predict_data['semantic_seg']
    
  # a0 = a[idx]
  # a_gt0 = a_gt[idx]
  # ( a0.shape, a_gt0.shape )

  res0 = model.predict ( np.expand_dims(np.expand_dims(a0,axis=0),axis=3) ) 
  res0 = res0.squeeze()
  res0_final = np.argmax(res0,axis=2)


  f, axarr = plt.subplots(1,3,figsize=(20,20))

  axarr[0].imshow(a0.squeeze(),cmap='gray_r')
  axarr[0].set_title( f'Echo {os.path.basename(model_val_data[batch_idx])}') #.set_text

  axarr[1].imshow(a_gt0.squeeze(),cmap=custom_cm) # gt
  axarr[1].set_title( 'Ground truth') #.set_text

  axarr[2].imshow(res0_final, cmap=custom_cm )
  axarr[2].set_title('Prediction')




#### Compute layer thickness (Maybe get raster too)
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

























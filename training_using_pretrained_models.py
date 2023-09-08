# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 22:20:38 2022

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

#from keras_segmentation.pretrained import pspnet_50_ADE_20K , pspnet_101_cityscapes, pspnet_101_voc12 #from keras_segmentation.models.model_utils import transfer_weights
# from keras_segmentation.models.segnet import resnet50_segnet


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
    wandb.init( project="my-test-project", entity="ibksolar", name='pretraining_model'+time_stamp,config ={})
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
base_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data'  # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'
aug_train_path = os.path.join(base_path,'augmented_plus_train_data\*.mat')
train_path = os.path.join(base_path,'train_data\*.mat')
val_path = os.path.join(base_path,'val_data\*.mat')
test_path = os.path.join(base_path,'test_data\*.mat')   

# Create tf.data.Dataset
config['batch_size'] = 2

config['img_y'] = 1664
config['img_x'] = 256

# Training params
#config={}
config['img_channels'] = 1

config['num_classes'] = 1
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

        layer = tf.cast( tf.cast(mat_file['raster'], dtype=tf.bool), dtype=tf.float64)        
        layer = tf.expand_dims(layer, axis=-1)
        
        shape0 = echo.shape        
        return echo,layer,np.asarray(shape0)     
    
    output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double, tf.int64])
    shape = output[2]
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([config['img_y'],config['img_x'], config['img_channels'] ])
    
    data1 = output[1]   
    data1.set_shape([config['img_y'],config['img_x'],config['num_classes']]) #,30, ,config['num_classes']    
    return data0,data1

train_ds = tf.data.Dataset.list_files(aug_train_path,shuffle=True) #'*.mat'
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



# ====================================================================
### Model Hyper params
# ====================================================================
config['dropout_rate'] = 0.1
config['num_heads'] = 12

num_transformer_layers = 8
hidden_units = [512,256]

num_patches = config['img_y']
embed_dim = config['img_x']

# ====================================================================
### Model Functions 
# ====================================================================

def mlp(x, hidden_units, dropout_rate = config['dropout_rate']):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.leaky_relu,)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
    
class FlippedPatchEncoder(layers.Layer):
    def __init__(self, num_patches, embed_dim, **kwargs):
        super(FlippedPatchEncoder, self).__init__(**kwargs)
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
                "num_patches": num_patches,
                "embed_dim": embed_dim,
            }
        )
        return 
    
    
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches , embed_dim , **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches        
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=embed_dim
        )

    def call(self, patch):
        
        # positions = tf.range(start=0, limit=self.num_patches, delta=1)
        # positions = self.position_embedding(positions)
        # # encoded = patch + tf.transpose(positions,perm= [0,1])
        # encoded = patch + positions
        
        patch = tf.cast( tf.transpose(patch,perm=[0,2,1]), dtype=tf.float64)
        
        positions = tf.range(start=0, limit=self.num_patches, delta=1,dtype=tf.float64)
        encoded = patch + tf.cast( self.position_embedding(positions), dtype=tf.float64 )
        
        encoded = tf.cast( tf.transpose(encoded,perm=[0,2,1]), dtype=tf.float64)      
        
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

# ====================================================================
# Model
# ====================================================================

# input_shape = (None, None, config['img_channels']) 
input_shape = (config['img_y'], config['img_x'],config['img_channels'])      
inputs = tf.keras.layers.Input(shape = input_shape, dtype=tf.float64)


# ====================================================================
# Attention
# ====================================================================

attn_large  = tf.keras.layers.Conv2D(16, (config['img_y'], 5), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
attn_large  = tf.keras.layers.Conv2D(1, (config['img_y'], 1), activation='relu', kernel_initializer='he_normal', padding='same')(attn_large)

attn_large2 = tf.squeeze(attn_large, axis = -1)

encoded = attn_large2  #PatchEncoder(config['img_x'],config['img_y'] ) (attn_large2) # Check this again


# Create multiple layers of the Transformer block.
for _ in range(num_transformer_layers):
    
    final_shape = [encoded.shape[-1]]
    # Layer normalization 1.
    x1 = layers.LayerNormalization(epsilon=1e-6)(encoded)
    
    # Create a multi-head attention layer.
    attention_output = layers.MultiHeadAttention(
        num_heads=config['num_heads'], key_dim=config['img_x'], dropout=0.1
    )(x1, x1)    
    
    # Skip connection 1.
    x2 = layers.Add()([attention_output, encoded])
    
    # Layer normalization 2.
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2) 
    
    # MLP
    mlp_out = mlp(x3,hidden_units+final_shape)
    
    # Skip connection 3.
    encoded = layers.Add()([x3, mlp_out])

encoded_patches = tf.expand_dims(encoded, axis = -1)    


# ====================================================================
# ResNet 50    
# ====================================================================
ResNet_50_model = sm.Unet(backbone_name='resnet50', encoder_weights='imagenet', encoder_freeze=True)
c1 = tf.keras.layers.Conv2D(3, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
ResNet_50_out = ResNet_50_model(c1)

# ====================================================================
# PSPNet
# ====================================================================
# Segnet_50_model = tf.keras.Sequential()
# Segnet_50_model.add( resnet50_segnet(n_classes = config['num_classes'], input_height=config['img_y'], input_width=config['img_x'] ) )
# Segnet_50_model.add( layers.Reshape(( config['img_y']//4, config['img_x'],1)) )
# Segnet_50_model.add( layers.Conv2DTranspose(config['img_y'], 3, strides = (2,1), padding='same') )
# Segnet_50_model.add( layers.Conv2DTranspose(config['img_y'], 3, strides = (2,1), padding='same') )
    
# PSP_50_out = Segnet_50_model(c1)
# PSP_50_out =  layers.Conv2D(1, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(PSP_50_out)

# strategy = tf.distribute.MirroredStrategy( cross_device_ops=tf.distribute.HierarchicalCopyAllReduce() )
# #print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
# with strategy.scope(): 

attn_out = layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same', activation=tf.nn.leaky_relu)(encoded_patches)


combined = attn_out + ResNet_50_out #PSP_50_out +
combined = tf.keras.layers.BatchNormalization()(combined) 
combined = tf.keras.layers.Conv2D(1, (3, 3), activation=tf.nn.leaky_relu, kernel_initializer='he_normal', padding='same', dtype=tf.float64)(combined)
combined = tf.keras.layers.Dropout(0.2)(combined)
combined = tf.keras.layers.BatchNormalization()(combined)
combined = tf.keras.layers.Activation('sigmoid')(combined)

# Sigmoid head to the outputs
attn_final = layers.Conv2D(1, (3, 3), kernel_initializer='glorot_normal', padding='same', activation='sigmoid', dtype=tf.float64)(attn_out)
ResNet_50_final = layers.Conv2D(1, (3, 3), kernel_initializer='glorot_normal', padding='same', activation='sigmoid', dtype=tf.float64)(ResNet_50_out)


model = tf.keras.Model(inputs,[combined,attn_final,ResNet_50_final]) #,PSP_50_out   

opt = keras.optimizers.Adam(learning_rate=config['learning_rate'])
opt2 = tfa.optimizers.AdamW(weight_decay = 0.001, learning_rate = config['learning_rate'])

loss = sm.losses.BinaryFocalLoss() #"binary_crossentropy"

model.compile(optimizer= opt, loss= loss, loss_weights=[0.4,0.3,0.3], metrics=['accuracy', sm.metrics.Precision(), sm.metrics.iou_score]) # sparse_categorical_crossentropy,jaccard_distance,binary_crossentropy,tf.keras.losses.KLDivergence(),

config['base_path'] = base_path
config['start_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
logz= f"{config['base_path']}/pretraining_model//{config['start_time']}_logs/"
callbacks = [
   ModelCheckpoint(f"{config['base_path']}//pretraining_model//pretraining_model_Checkpoint{time_stamp}.h5", save_best_only=True, monitor="val_loss"),
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
model.save(f"{config['base_path']}//pretraining_model//pretraining_model_acc_{acc:.2f}_GOOD_{time_stamp}.h5")


model_val_data_path = os.path.join(base_path,'test_data\*.mat')
model_val_data = glob.glob(model_val_data_path)


## Final prediction correcting code
import itertools
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


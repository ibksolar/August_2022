# -*- coding: utf-8 -*-
"""
Created 3rd June,2022

@author: i368o351
"""

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
from custom_loadmat import loadmat
from scipy.ndimage import median_filter as sc_med_filt

# from keras.metrics import MeanIoU
# from sklearn.metrics import roc_auc_score

import glob
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,ReduceLROnPlateau
from datetime import datetime

from focal_loss import SparseCategoricalFocalLoss

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


use_wandb = False
time_stamp = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')

if use_wandb:    
    ## WandB config
    import wandb
    from wandb.keras import WandbCallback    
    
    wandb.init( project="my-test-project", entity="ibksolar", name='EchoViT1_new'+time_stamp,config ={})
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



# PATHS
# Path to data
base_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\SR_Dataset_v1\Dec'  # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'
#train_aug = os.path.join(base_path,'augmented_plus_train_data\*.mat')
train_path = os.path.join(base_path,'train_data\*.mat')
val_path = os.path.join(base_path,'val_data\*.mat')
test_path = os.path.join(base_path,'test_data\*.mat')   

# Create tf.data.Dataset
config['Run_Note'] = 'First try of EchoViT1 with SMB and Elevation'
config['batch_size'] = 8

# config['img_y'] = 416 #1664 #1664 , 416
# config['img_x'] = 256 #256, 64
config['img_y'],config['img_x'] = loadmat(glob.glob(train_path)[0])['echo_tmp'].shape




# Training params
#config={}
config['img_channels'] = 1

config['num_classes'] = 31
config['epochs'] = 200
config['learning_rate'] = 1e-4
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
        
        #if tf.random.uniform(())> 0.1:
        aug_type = tf.random.uniform((1,1),minval=1, maxval=8,dtype=tf.int64).numpy()      
 
        # if aug_type == 1:
        #     echo = tf.experimental.numpy.fliplr(echo)
        #     layer = tf.experimental.numpy.fliplr(layer)
        
        if aug_type == 1: # Constant offset
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
            
        # elif aug_type == 7: # Random brightness
        #     echo = tf.image.random_saturation(echo, 0.1, 0.9)
        #     echo = tf.clip_by_value(echo, 0, 1)
        
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
    data1.set_shape([config['img_y'],config['img_x'],config['img_channels']]) #,30, ,config['num_classes']    
    return data0,data1


def read_mat_multiple(filepath):
    def _read_mat(filepath):
        
        filepath = bytes.decode(filepath.numpy())      
        mat_file = loadmat(filepath)
        echo = tf.cast(mat_file['echo_tmp'], dtype=tf.float64)
        
        smb = tf.cast(np.mean(mat_file['weather_data']['curr_smb'],axis=0), dtype=tf.float64)
        elev = tf.cast(mat_file['new_Elev'], dtype=tf.float64)         
        
        echo = tf.expand_dims(echo, axis=-1)
        
        if config['img_channels']> 1:            
            echo = tf.image.grayscale_to_rgb(echo)          
        
        if config['num_classes'] == 1:
            layer = tf.cast( tf.cast(mat_file['raster'], dtype=tf.bool), dtype=tf.float64)        
            layer = tf.expand_dims(layer, axis=-1)
        else:
            layer = tf.cast(mat_file['semantic_seg'], dtype=tf.float64) 
            layer = tf.expand_dims(layer, axis=-1)
            #layer = tf.keras.utils.to_categorical(layer, config['num_classes'] )
        
        shape0 = echo.shape        
        return echo,smb,elev,layer,np.asarray(shape0)     
    
    output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double,tf.double, tf.double, tf.int64])
    shape = output[4]
    
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([config['img_y'],config['img_x'], config['img_channels'] ])
    
    data1 = output[1] 
    data1.set_shape([ config['img_x']])
    
    data2 = output[2] 
    data2.set_shape([config['img_x']])
    
    data3 = output[3]   
    data3.set_shape([config['img_y'],config['img_x'],config['img_channels']]) #,30, ,config['num_classes']    
    return {'echo':data0,'smb':data1, 'elev':data2},data3

train_ds = tf.data.Dataset.list_files(train_path,shuffle=True) #'*.mat'
train_ds = train_ds.map(read_mat_multiple,num_parallel_calls=8)
train_ds = train_ds.batch(config['batch_size'],drop_remainder=True).prefetch(AUTO) #.shuffle(buffer_size = 100 * config['batch_size'])

# No augmentation for testing and validation
val_ds = tf.data.Dataset.list_files(val_path,shuffle=True)
val_ds = val_ds.map(read_mat_multiple,num_parallel_calls=8)
val_ds = val_ds.batch(config['batch_size'],drop_remainder=True).cache().prefetch(AUTO)

test_ds = tf.data.Dataset.list_files(test_path,shuffle=True)
test_ds = test_ds.map(read_mat_multiple,num_parallel_calls=8)
test_ds = test_ds.batch(config['batch_size'],drop_remainder=True).cache().prefetch(AUTO)

# train_shape = [ ( tf.shape(item[0]).numpy(),tf.shape(item[1]).numpy() ) for item in train_ds.take(1) ]
# train_shape = train_shape[0]

# print(f' X_train train shape {train_shape[0]}')
# print(f' Training target shape {train_shape[1]}')


config['Row_embed_dim'] = config['Col_num_patches'] = config['img_x'] #64   
config['Row_num_patches'] = config['Col_embed_dim'] = config['img_y'] #416




# =========================================================================================================== 
# MLP
# ===========================================================================================================  
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation= tf.nn.gelu) (x)
        x = layers.Dropout(dropout_rate)(x)
    return x


# =========================================================================================================== 
# RowPatch : return sum of embedded patch and postion embedding 
# ===========================================================================================================  
class RowPatchEncoder(layers.Layer):
    def __init__(self, num_patches = config['Row_num_patches'], embed_dim = config['Row_embed_dim'], **kwargs):
        super(RowPatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=embed_dim)
        self.position_embedding = layers.Embedding(
            input_dim=1, output_dim=embed_dim #, input_length = num_patches
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
                "num_patches": config['Row_num_patches'] ,
                "embed_dim": config['Row_embed_dim'],
            }
        )
        return config
# ===========================================================================================================


# =========================================================================================================== 
# ColPatch : only return inverted position embedding  
# ===========================================================================================================    
class ColPatchEncoder(layers.Layer):
    def __init__(self, num_patches = config['Col_num_patches'], embed_dim = config['Col_embed_dim'], **kwargs):
        super(ColPatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        
        self.position_embedding = layers.Embedding(
            input_dim=1, output_dim=embed_dim, input_length = num_patches
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = patch+ tf.transpose( self.position_embedding(positions), perm=[1,0] )
        return encoded

    # Override function to avoid error while saving model
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {               
                "num_patches": config['Col_num_patches'],
                "embed_dim": config['Col_embed_dim'],
            }
        )
        return config    
# ===========================================================================================================
    



# Losses
alpha = 0.1
gamma = 3

class FocalLoss(tf.losses.Loss):
    """Implements Focal loss"""

    def __init__(self, alpha = alpha, gamma = gamma,**kwargs):
        super(FocalLoss, self).__init__(
            reduction="none", name="RetinaNetFocalLoss"
        )
        self._alpha = tf.cast(alpha,tf.float64)
        self._gamma = tf.cast(gamma, tf.float64)

    def call(self, y_true, y_pred):
        
        y_true = tf.cast(y_true, tf.float64)
        y_pred = tf.cast(y_pred, tf.float64)
        
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred,
        )
        probs = tf.nn.sigmoid(y_pred)
        alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)
    
    # def get_config(self):
    #     config = super().get_config().copy()
    #     config.update({            
    #         'alpha' : self._alpha,
    #         'gamma' : self._gamma,
            
    #         })
    #     return config
custom_loss = FocalLoss(alpha = alpha, gamma =gamma )



import tensorflow.keras.backend as K

def dice_coef(y_true, y_pred, smooth=1e-7, num_classes = config['num_classes']):
   
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes = num_classes)[Ellipsis,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    
    y_true_f = K.cast(y_true_f, 'float32')
    y_pred_f = K.cast(y_pred_f, 'float32')
    
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


# Model hyper-param
num_heads = 12
dense_dim = 512

gen_embed_dim = config['Col_embed_dim']
proj_head_units = [1024, 512, 64] #2048,
mlp_head_units = [ 512,256] #mlp_head_units = [ 2048,1024, 512, 64]  # Size of the dense layers , 2048, 1024,

# Project inputs
dense_proj = tf.keras.Sequential()
for units in proj_head_units:
    dense_proj.add(layers.Dense(units, activation='relu', use_bias=True) )

transformer_layers = 12

kernel_init = 'glorot_normal'

input_shape = (config['img_y'], config['img_x'], config['img_channels'])
meta_input_shape = (config['img_x'],)

dtype_used = tf.float64

#strategy = tf.distribute.MirroredStrategy( cross_device_ops=tf.distribute.HierarchicalCopyAllReduce() )
#print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

#with strategy.scope():    

inputs = layers.Input(shape=input_shape, dtype=dtype_used, name='echo')    
input2 = layers.Input(shape = meta_input_shape, dtype=dtype_used, name='smb') #name='smb'
input3 = layers.Input(shape = meta_input_shape, dtype=dtype_used, name='elev') #name='elev'

input2 = tf.cast( layers.BatchNormalization()(input2), dtype= dtype_used) # SMB
input3 = tf.cast(  layers.BatchNormalization()(input3) , dtype= dtype_used)  # Elev

# ====================================================================
# ResNet 50    
# ====================================================================
ResNet_50_model = sm.Unet(backbone_name='resnet50', encoder_weights='imagenet', encoder_freeze=True)
in1 =  layers.Conv2D(3, (3, 3), activation='relu', kernel_initializer= kernel_init, padding='same')(inputs)
in1 = tf.cast( ResNet_50_model(in1) , dtype= dtype_used)

in2 = layers.Conv2D(1, (config['img_y'],1), activation='relu', kernel_initializer= kernel_init, padding='same' , dtype= dtype_used )(in1)

## Col Encode patches
x = tf.reduce_mean(in2,axis = -1)
Col_encoded_patches = x #tf.cast( x + pos_embed , dtype=tf.float32)
# Col_pos_embd = layers.Embedding(input_dim=2, output_dim = config['img_y'], input_length = config['img_x'])(tf.range(start=0, limit=config['img_x'], delta=1))
# Col_pos_embd = layers.Reshape(input_shape[:2])(Col_pos_embd)
# Col_encoded_patches = Col_encoded_patches + Col_pos_embd

## Row Encode patches    
Row_encoded_patches = layers.Dense(units = config['img_x'])(x)
Row_pos_embd = layers.Embedding(input_dim=2, output_dim = config['img_x'], input_length = config['img_y'])(tf.range(start=0, limit=config['img_y'], delta=1))
Row_encoded_patches = tf.cast(Row_encoded_patches + Row_pos_embd, dtype = dtype_used)

## SMB embed  
smb_embed = layers.Embedding(input_dim =2, output_dim = config['img_y'], input_length = config['img_x'])(input2)
smb_embed = tf.cast( layers.Reshape(input_shape[:2] )(smb_embed) , dtype = dtype_used)

## Elev embed  
elev_embed = layers.Embedding(input_dim =2, output_dim = config['img_y'], input_length = config['img_x'])(input3)
elev_embed = tf.cast( layers.Reshape(input_shape[:2] )(elev_embed) , dtype = dtype_used)

encoded_patches = encoded_patches2 = Row_encoded_patches + Col_encoded_patches + smb_embed + elev_embed # Combined

# Create multiple layers of the Transformer block.
for _ in range(transformer_layers):
    
    final_shape = [encoded_patches.shape[-1] ]
    
    # Layer normalization 1.
    x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    # Create a multi-head attention layer.
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=gen_embed_dim, dropout=0.1
    )(x1, x1)
    
    # Skip connection 1.
    x2 = layers.Add()([attention_output, encoded_patches])
    # Layer normalization 2.
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    
    # Skip connection 2.
    # mlp_in = layers.Add()([x3, x2])    
    # # # # MLP (Newly Added: Might delete)
    # mlp_out = mlp(mlp_in, mlp_head_units+final_shape ,dropout_rate = 0.2)  
    
    # Skip connection 2.
    encoded_patches = x3 #layers.Add()([x3, mlp_out])
    # # # MLP (Newly Added: Might delete)

# Project Transformer output     
#representation = dense_proj(encoded_patches)

# Skip connection
encoded_patches = tf.cast(encoded_patches, dtype = dtype_used) + encoded_patches2

# Create a [batch_size, embed_dim] tensor.
#representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
encoded_patches_rep = tf.expand_dims(encoded_patches, axis=-1)  
Col_encoded_patches = tf.expand_dims(Col_encoded_patches, axis=-1)  
Row_encoded_patches = tf.expand_dims(Row_encoded_patches, axis=-1)  

used_activation = 'sigmoid' if config['num_classes'] == 1 else 'softmax'

Col_embed_side =  layers.Conv2D(config['num_classes']*5, (1,1), activation='relu', padding="same")(Col_encoded_patches)
Col_embed_side_out =  layers.Conv2D(config['num_classes'], (1,1),  dtype = dtype_used, activation = used_activation, padding="same", name ='ColEmbed')(Col_embed_side)

Row_embed_side =  layers.Conv2D(config['num_classes']*5, (1,1), activation='relu', padding="same")(Row_encoded_patches)
Row_embed_side_out =  layers.Conv2D(config['num_classes'], (1,1),  dtype = dtype_used, activation = used_activation, padding="same", name ='RowEmbed')(Row_embed_side)

all_comb_ouput = encoded_patches_rep + Col_encoded_patches + Row_encoded_patches

encoded_patches_rep = layers.Conv2D(config['num_classes']*5, (1,1), activation='relu', padding="same", dtype = dtype_used)(encoded_patches_rep)
encoded_patches_rep = layers.Conv2D(config['num_classes'], (1,1), activation="relu", padding="same", dtype = dtype_used )(encoded_patches_rep)   # activity_regularizer='l2', kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4), 
layers.Dropout(0.3)(encoded_patches_rep)
encoded_patches_output = layers.Conv2D(config['num_classes'], (1,1),  dtype = dtype_used, activation = used_activation, padding="same", name = 'encoded_patches')(encoded_patches_rep) #activity_regularizer='l2',


all_comb_ouput =  layers.Conv2D(config['num_classes'], (1,1),  dtype = dtype_used, activation = used_activation, padding="same", name='Combined')(all_comb_ouput)


# return Keras model.
model = keras.Model(inputs={"echo":inputs,"smb":input2,"elev":input3}, outputs=[encoded_patches_output,Col_embed_side_out,Row_embed_side_out,all_comb_ouput])

#model =  create_vit_object_detector(input_shape,num_patches,embed_dim,num_heads,transformer_units,transformer_layers,mlp_head_units,)

opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate']) #, clipnorm=0.5
opt2 = tfa.optimizers.AdamW(weight_decay = 0.001, learning_rate=config['learning_rate'])

## Print Start time
config['start_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
print(f' Start time {config["start_time"]}')


logz= f"{config['base_path']}/echo_vit_smb/{config['start_time']}_logs/"
callbacks = [
   ModelCheckpoint(f"{config['base_path']}//echo_vit_smb//echo_vit_smb_Checkpoint{time_stamp}.h5", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.25, patience=10, min_lr=0.00005, verbose= 1),
    EarlyStopping(monitor="val_loss", patience=30, verbose=1), 
    TensorBoard(log_dir = logz,histogram_freq = 1,profile_batch = '1,70', embeddings_freq=50),
    #WandbCallback()
]

loss = tf.keras.losses.BinaryCrossentropy() if config['num_classes']==1 else tf.keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer=opt,#opt
          loss= loss, #SparseCategoricalFocalLoss(gamma=2), #tf.keras.losses.CategoricalCrossentropy(), #custom_loss, dice_coef_loss tf.keras.losses.CategoricalCrossentropy()
          metrics=['accuracy']) #,tf.keras.metrics.MeanIoU(num_classes, name="MeanIoU")

history = model.fit(train_ds, epochs = config['epochs'],validation_data = val_ds, callbacks = callbacks)

## Print End time
config['start_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
print(f' Completion time {config["start_time"]}')



loaded_model = tf.keras.models.load_model(f"{config['base_path']}//echo_vit_smb//echo_vit_smb_Checkpoint{time_stamp}.h5",
                                   custom_objects={"RowPatchEncoder":RowPatchEncoder,"ColPatchEncoder":ColPatchEncoder,"FocalLoss":FocalLoss})

        

# Save model with proper name
_,acc = model.evaluate(test_ds)
model.save(f"{config['base_path']}//echo_vit_smb//echo_vit_smb{acc:.2f}_{time_stamp}.h5")

# Custom colormap
custom_cm = cm.BuPu(np.linspace(0,1,30))
custom_cm = colors.ListedColormap(custom_cm[10:,:-1])

## Visualize result of model prediction for "unseen" echogram during training
model_val_data_path = os.path.join(base_path,'new_test\image\*.mat')
model_val_data = glob.glob(model_val_data_path)

batch_idx = random.randint(1,len(model_val_data)) # Pick any of the default batch

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



# # Predict on the entire test data
# test_pred = model.predict(x_test)
# test_pred = np.argmax(test_pred,axis =3)

# from sklearn.metrics import classification_report
# print( classification_report( y_test.flatten(),test_pred.flatten(), labels=list(range(num_classes)), zero_division=1 ))


# IoU = MeanIoU(num_classes = num_classes)
# IoU.update_state(y_test.flatten(), test_pred.flatten() );
# MIoU = IoU.result().numpy()
# print(f'Mean IoU is {100*MIoU:.2f}%')



# #roc_auc_score(y_test.flatten(), test_pred.flatten(),multi_class='OvR', labels=list(range(30)) )
















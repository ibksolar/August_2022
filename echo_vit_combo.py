# -*- coding: utf-8 -*-
"""
Created 3rd June,2022

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

from keras.metrics import MeanIoU
from sklearn.metrics import roc_auc_score

import glob
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,ReduceLROnPlateau
import segmentation_models as sm
from datetime import datetime

from focal_loss import SparseCategoricalFocalLoss

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
    
    wandb.init( project="my-test-project", entity="ibksolar", name='EchoViT1'+time_stamp,config ={})
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
base_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\new_trainJuly'  # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'
train_path = os.path.join(base_path,'train_data\*.mat')
val_path = os.path.join(base_path,'val_data1\*.mat')
test_path = os.path.join(base_path,'test_data\*.mat')   

# Create tf.data.Dataset
config['Run_Note'] = 'Trying combined fast time and slow time embed'
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
            echo = echo + tf.random.normal(shape=(416,64,config['img_channels']),stddev=0.5,dtype=tf.float64)
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
            echo = echo - tf.random.normal(shape=(416,64,config['img_channels']),stddev=0.5,dtype=tf.float64)
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
        # row_idx,_ = tf.unique(tf.reshape(tf.random.uniform((1,20),minval=1, maxval=416,dtype=tf.int64),[10]) )
        # echo[row_idx,:] = tf.Variable(0, dtype=tf.float64)
        

        #echo = tf.experimental.numpy.flipud(echo)
        #layer = tf.experimental.numpy.flipud(layer)                            
                   
        
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
        
        layer = tf.expand_dims(layer, axis=-1)
        #layer = tf.keras.utils.to_categorical(layer, config['num_classes'] )
        shape0 = echo.shape #mat_file['echo_tmp'].shape
        
        return echo,layer,np.asarray(shape0)
    output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double, tf.int64])
    shape = output[2]
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([416,64,config['img_channels']])
    
    data1 = output[1]   
    data1.set_shape([416,64,1])#,30    
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
        #layer = tf.keras.utils.to_categorical(layer, config['num_classes'] )
        shape0 = echo.shape #mat_file['echo_tmp'].shape        
        return echo,layer,np.asarray(shape0)     
    
    output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double, tf.int64])
    shape = output[2]
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([416,64,config['img_channels']])
    
    data1 = output[1]   
    data1.set_shape([416,64,1]) #,30   
    return data0,data1

train_ds = tf.data.Dataset.list_files(train_path,shuffle=True) #'*.mat'
train_ds = train_ds.map(read_mat_train,num_parallel_calls=8)
train_ds = train_ds.batch(config['batch_size'],drop_remainder=True).prefetch(AUTO) #.shuffle(buffer_size = 100 * config['batch_size'])

# No augmentation for testing and validation
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



# Model hyper-param
config['embed_dim'] = embed_dim =  train_shape[0][2] #416
config['num_patches'] = num_patches = train_shape[0][1] #64 

config['embed_dim_flipped'] = train_shape[0][1]
config['num_patches_flipped'] = train_shape[0][2]

num_heads = 20
dense_dim = 512

transformer_units = [
    embed_dim * 4,
    embed_dim,
]

transformer_layers = 20

proj_head_units = [1024, 512, 64] #2048,
mlp_head_units = [ 512,256,64] 
#mlp_head_units = [ 2048,1024, 512, 64]  # Size of the dense layers , 2048, 1024,

#input_shape = (416, 64)
input_shape = (416, 64, config['img_channels'])


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches = config['num_patches'], embed_dim = config['embed_dim'], **kwargs):
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

class FlippedPatchEncoder(layers.Layer):
    def __init__(self, num_patches2 , embed_dim2 , **kwargs):
        super(FlippedPatchEncoder, self).__init__(**kwargs)
        self.embed_dim2 = embed_dim2
        self.num_patches2 = num_patches2
        self.projection = layers.Dense(units=embed_dim2)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches2, output_dim=embed_dim2
        )

    def call(self, patch):
        patch = tf.transpose(patch,perm=[0,2,1])
        
        positions = tf.range(start=0, limit=self.num_patches2, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        
        encoded = tf.transpose(encoded,perm=[0,2,1])
        return encoded

    # Override function to avoid error while saving model
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {               
                "num_patches2": self.num_patches2,
                "embed_dim2": self.embed_dim2,
            }
        )
        return config
    
    
# Losses
alpha = 0.25
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


# def create_vit_object_detector(
#     input_shape,   
#     num_patches,
#     embed_dim,
#     num_heads,
#     transformer_units,
#     transformer_layers,
#     mlp_head_units,
# ):
    
inputs = layers.Input(shape=input_shape)

# Project inputs
dense_proj = tf.keras.Sequential()
for units in proj_head_units:
    dense_proj.add(layers.Dense(units, activation='relu', use_bias=True) )
    

# Encode patches
x = tf.reduce_mean(inputs,axis = -1)

encoded_patches = PatchEncoder(num_patches, embed_dim)(x)
encoded_patches_flipped = FlippedPatchEncoder( config['num_patches_flipped'], config['embed_dim_flipped'])(x) # this is flipped on purpose

encoded_final = encoded_patches + encoded_patches_flipped

encoded_patches = dense_proj(encoded_final) + encoded_final

# Create multiple layers of the Transformer block.
for _ in range(transformer_layers):
    # Layer normalization 1.
    x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    # Create a multi-head attention layer.
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim, dropout=0.1
    )(x1, x1)
    # Skip connection 1.
    x2 = layers.Add()([attention_output, encoded_patches])
    # Layer normalization 2.
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    
    # Skip connection 2.
    encoded_patches = layers.Add()([x3, x2])
    
    # # # MLP (Newly Added: Might delete)
    # mlp_out = mlp(encoded_patches,mlp_head_units,dropout_rate = 0.1)    
    # # Skip connection 2.
    # encoded_patches = layers.Add()([x3, mlp_out])
    # # # MLP (Newly Added: Might delete)

# Project Transformer output     
representation = dense_proj(encoded_patches)

# Create a [batch_size, embed_dim] tensor.
representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

representation= tf.expand_dims(representation, axis=-1)  

representation = layers.Conv2D(config['num_classes']*5, (7,5), activation=tf.nn.gelu, padding="same")(representation)
representation = layers.Conv2D(config['num_classes']*5, 3, activation='relu', padding="same")(representation)
representation = layers.Conv2D(config['num_classes'], 3, activation="relu", padding="same", )(representation)   # activity_regularizer='l2', kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4), 
layers.Dropout(0.3)(representation)
output = layers.Conv2D(config['num_classes'], (1,1),  dtype = tf.float32, padding="same")(representation) #activity_regularizer='l2', activation="softmax"
 

# return Keras model.
model = keras.Model(inputs=inputs, outputs=output)

#model =  create_vit_object_detector(input_shape,num_patches,embed_dim,num_heads,transformer_units,transformer_layers,mlp_head_units,)

opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate']) #, clipnorm=0.5
opt2 = tfa.optimizers.AdamW(weight_decay = 0.001, learning_rate=config['learning_rate'])

## Print Start time
config['start_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
print(f' Start time {config["start_time"]}')


logz= f"{config['base_path']}/echo_vit/{config['start_time']}_logs/"
callbacks = [
   ModelCheckpoint(f"{config['base_path']}//echo_vit_combo//echo_vit_combo_Checkpoint{time_stamp}.h5", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=0.000005, verbose= 1),
    EarlyStopping(monitor="val_loss", patience=30, verbose=1), 
    TensorBoard(log_dir = logz,histogram_freq = 1,profile_batch = '1,70', embeddings_freq=50),
    WandbCallback()
]

loss = SparseCategoricalFocalLoss(gamma = 3, from_logits = True) #tf.keras.losses.CategoricalCrossentropy()
model.compile(optimizer=opt,
          loss= loss, #tf.keras.losses.CategoricalCrossentropy(), #custom_loss, dice_coef_loss tf.keras.losses.CategoricalCrossentropy()
          metrics=['accuracy']) #,tf.keras.metrics.MeanIoU(num_classes, name="MeanIoU")

history = model.fit(train_ds, epochs = config['epochs'],validation_data = val_ds, callbacks = callbacks)

## Print End time
config['start_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
print(f' Completion time {config["start_time"]}')


if type(loss) is not keras.losses.CategoricalCrossentropy:
    loaded_model = tf.keras.models.load_model(f"{config['base_path']}//echo_vit//echo_vit_Checkpoint{time_stamp}.h5",
                                   custom_objects={"PatchEncoder":PatchEncoder,"FocalLoss":FocalLoss})
else:
    loaded_model = tf.keras.models.load_model(f"{config['base_path']}//echo_vit//echo_vit_Checkpoint{time_stamp}.h5",
                                       custom_objects={"PatchEncoder":PatchEncoder})
        

# Save model with proper name
_,acc = model.evaluate(test_ds)
model.save(f"{config['base_path']}//echo_vit_combo//echo_vit_combo{acc:.2f}_{time_stamp}.h5")

# Custom colormap
custom_cm = cm.BuPu(np.linspace(0,1,30))
custom_cm = colors.ListedColormap(custom_cm[10:,:-1])

## Visualize result of model prediction for "unseen" echogram during training
model_val_data_path = os.path.join(base_path,'short_eval_data\*.mat')
#model_val_data_path = os.path.join(base_path,'new_test\image\*.mat')
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
















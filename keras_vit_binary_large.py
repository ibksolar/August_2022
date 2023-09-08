# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 10:16:35 2022
Used for Class Project and not edited afterwards (except for adding Colormap and thickness code)

Updated: Nov24th, 2022 (ThanksGiving Night)

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

#import albumentations as A

import glob
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,ReduceLROnPlateau

import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()

from datetime import datetime

# from albumentations import (
#     Compose, RandomBrightness, RandomContrast, VerticalFlip,ElasticTransform,GaussianBlur,RandomBrightnessContrast
    
# )
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
time_stamp = '24_November_22_2306' #datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')

if use_wandb:    
    ## WandB config
    import wandb
    from wandb.keras import WandbCallback    
    
    wandb.init( project="my-test-project", entity="ibksolar", name='KerasViT_large_binary'+time_stamp, config = {} )
    config = wandb.config
else:
    config ={}
    
    
time_stamp = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')


   
    
     
    
# ==================LOAD DATA =========================================
#========================================================================

# PATHS
# Path to data
base_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data'  # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'
train_path = os.path.join(base_path,'train_data\*.mat')
val_path = os.path.join(base_path,'val_data\*.mat')
test_path = os.path.join(base_path,'test_data\*.mat')   

# Hyperparameters
config['Run_Note'] = 'Old Keras Vit for large binary segmentation_fixed train_ds bad augmentation issue'
config['batch_size'] = 4


# Training params
config['img_y'] = 416*4
config['img_x'] = 64*4

config['img_channels'] = 3
config['weight_decay'] = 0.0001

config['num_classes'] = 1 #30
config['epochs'] = 500
config['learning_rate'] = 1e-3
config['base_path'] = base_path
SEED = 42
AUTO = tf.data.experimental.AUTOTUNE


# =============================================================================
# Manual augmentation
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
    data0.set_shape([1664,256,config['img_channels']])
    
    data1 = output[1]   
    data1.set_shape([1664,256,1 ])#,30,config['num_classes']    
    return data0,data1


# =============================================================================

# transforms = Compose( [RandomBrightness(limit=0.1),RandomBrightnessContrast(brightness_limit=0.15,contrast_limit=0.15, p= 0.3),VerticalFlip(), ElasticTransform(alpha=0.1,sigma = 0.2)])


# ## Function for albumentation dataset    
# def read_mat_alb(filepath):
#     def _read_mat(filepath):
        
#         filepath = bytes.decode(filepath.numpy())      
#         mat_file = loadmat(filepath)
#         echo = tf.cast(mat_file['echo_tmp'], dtype=tf.float32)
        
#         echo = tf.expand_dims(echo, axis=-1)        
#         if config['img_channels'] > 1:
#             echo = tf.image.grayscale_to_rgb(echo)
        
#         # layer = tf.cast(mat_file['raster'], dtype=tf.float64) 
#         layer = tf.cast( tf.cast(mat_file['raster'], dtype=tf.bool), dtype=tf.float32)
        
#         layer = tf.expand_dims(layer, axis=-1)
        
#         transformed = transforms(image=echo.numpy(), mask=layer.numpy() )
        
#         echo = transformed["image"]
#         layer = transformed["mask"]
        
#         # layer = tf.keras.utils.to_categorical(layer, config['num_classes'] )
#         shape0 = echo.shape #mat_file['echo_tmp'].shape        
#         return echo,layer,np.asarray(shape0)     
#     output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double, tf.int64])
#     shape = output[2]
#     data0 = tf.reshape(output[0], shape)
#     data0.set_shape([config['img_y'],config['img_x'],config['img_channels']])
    
#     data1 = output[1]   
#     data1.set_shape([config['img_y'],config['img_x'],1]) #,30   
#     return data0,data1

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

# Model hyper-param
config['embed_dim'] = embed_dim =  train_shape[0][2] #416
config['num_patches'] = num_patches = train_shape[0][1] #64 

config['embed_dim_flipped'] = train_shape[0][1]
config['num_patches_flipped'] = train_shape[0][2]
config['dropout_rate'] = 0.1

num_heads = 20
dense_dim = 512

transformer_units = [
    embed_dim * 4,
    embed_dim,
]

transformer_layers = 20

proj_head_units = [1024, 512, config['img_x']] #2048,
mlp_head_units = [ 512,256,config['img_x']] 

input_shape = (config['img_y'], config['img_x'], config['img_channels'])



def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches = num_patches, embed_dim = embed_dim, **kwargs):
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
            }
        )
        return config


# Losses
alpha = 0.1
gamma = 10

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


def Transformer(encoded_patches):
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
        
        # mlp
        x3 = mlp(x3, mlp_head_units, dropout_rate = 0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
        
        return encoded_patches

dtype_used = tf.float64 
   
inputs = layers.Input(shape=input_shape, dtype=dtype_used)

used_activation ='softmax' if config['num_classes'] > 1 else 'sigmoid'

## Project inputs
dense_proj = tf.keras.Sequential()
for units in mlp_head_units:
    dense_proj.add(layers.Dense(units, activation='relu', use_bias=False) )

## Position Embedding
pos_embed = layers.Embedding(input_dim = config['img_y'], output_dim = config['img_x'], input_length=config['img_y'] ) (tf.range(start=0, limit= config['img_y'])) 
pos_embed = tf.cast(pos_embed, dtype= dtype_used)

## Just Res50 branch
Res50 = sm.Unet(backbone_name='resnet50',encoder_weights='imagenet',  encoder_freeze=True)
Res50 = tf.cast( Res50(inputs),  dtype=dtype_used)


## Res50_attn => Res50 + attention branch

Res50_attn = layers.Conv2D(1,(config['img_y'],1), padding='same', activation='relu',  dtype=dtype_used )(Res50)
Res50_attn = tf.reduce_mean(Res50_attn,axis = -1)
Res50_attn = Res50_attn + pos_embed

Res50_attn = tf.cast( Transformer(Res50_attn),  dtype=dtype_used)
Res50_attn = tf.expand_dims(Res50_attn, axis=-1)
Res50_attn = layers.Conv2D(64, 1, activation='relu', padding="same", dtype= dtype_used)(Res50_attn)
Res50_attn = layers.Conv2D(1, 1, activation='relu', padding="same", dtype= dtype_used)(Res50_attn)

## Pure (just) attention branch
pure_attn = layers.Conv2D(1, (config['img_y'],1), padding='same', dtype= dtype_used)(inputs) # No activation
pure_attn = tf.reduce_mean(pure_attn,axis = -1)
pure_attn = pure_attn + pos_embed

pure_attn = tf.cast( Transformer(pure_attn) , dtype= dtype_used )
pure_attn = tf.expand_dims(pure_attn, axis=-1)
pure_attn = layers.Conv2D(64, 1, activation='relu', padding="same", dtype= dtype_used)(pure_attn)
pure_attn = layers.Conv2D(1, 1, activation='relu', padding="same", dtype= dtype_used)(pure_attn)

## Outputs
combined = Res50 + Res50_attn + pure_attn
combined = layers.Conv2D(64, 1,padding='same', activation ='relu', dtype= dtype_used )(combined)
combined = layers.Conv2D(1, 1,padding='same', activation = used_activation, name= 'combined_out', dtype= dtype_used)(combined)

Res50_out = layers.Conv2D(1, 1, activation = used_activation, padding="same", name= 'Res50_out', dtype= dtype_used)(Res50)
Res50_attn_out = layers.Conv2D(1, 1, activation = used_activation, padding="same", name= 'Res50_attn_out', dtype= dtype_used)(Res50_attn)
pure_attn_out = layers.Conv2D(1, 1, activation = used_activation, padding="same", name= 'pure_attn_out', dtype= dtype_used)(pure_attn)

    
#encoded_patches = dense_proj(encoded_patches)
# return Keras model.
model = keras.Model(inputs=inputs, outputs=[combined,Res50_out,Res50_attn_out,pure_attn_out])


opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate']) #, clipnorm=0.5
opt2 = tfa.optimizers.AdamW(weight_decay = 0.001, learning_rate=config['learning_rate'])

config['start_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
logz= f"{config['base_path']}/KerasViT_large_binary/{config['start_time']}_logs/"
callbacks = [
   ModelCheckpoint(f"{config['base_path']}//KerasViT_large_binary//KerasViT_large_binary_Checkpoint{time_stamp}.h5", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.25, patience=10, min_lr=0.00005, verbose= 1),
    EarlyStopping(monitor="val_loss", patience=30, verbose=1), 
    TensorBoard(log_dir = logz,histogram_freq = 1,profile_batch = '1,70', embeddings_freq=50),
    WandbCallback()
]

loss = sm.losses.BinaryFocalLoss() #tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer=opt,
          loss= loss, # [loss,custom_loss,sm.losses.DiceLoss()], #tf.keras.losses.CategoricalCrossentropy(), #custom_loss, dice_coef_loss tf.keras.losses.CategoricalCrossentropy()
          metrics=['accuracy']) #,tf.keras.metrics.MeanIoU(num_classes, name="MeanIoU"), ,sm.metrics.iou_score

history = model.fit(train_ds, epochs = config['epochs'],validation_data = val_ds, callbacks = callbacks)


if type(loss) is not keras.losses.CategoricalCrossentropy:
    loaded_model = tf.keras.models.load_model(f"{config['base_path2']}//KerasViT_large_binary//KerasViT_large_binary_Checkpoint{time_stamp}.h5",
                                   custom_objects={"PatchEncoder":PatchEncoder,"FocalLoss":FocalLoss,"iou_score":sm.metrics.iou_score})
else:
    loaded_model = tf.keras.models.load_model(f"{config['base_path2']}//KerasViT_large_binary//KerasViT_large_binary_Checkpoint{time_stamp}.h5",
                                       custom_objects={"PatchEncoder":PatchEncoder,"iou_score":sm.metrics.iou_score})
        


# Save model with proper name
_,acc = model.evaluate(test_ds)
model.save(f"{config['base_path']}//KerasViT_large_binary//KerasViT_large_binarySegmentation_{acc:.3f}_{time_stamp}.h5")


custom_cm = cm.Blues(np.linspace(0,1,30))
custom_cm = colors.ListedColormap(custom_cm[10:,:-1])


model_val_data_path = os.path.join(base_path,'test_data\*.mat')
model_val_data = glob.glob(model_val_data_path)

batch_idx = random.randint(1,len(model_val_data)) # Pick any of the default batch

for _ in range(10):
    batch_idx = random.randint(1,len(model_val_data) - 10) # Pick any of the default batch
    
    for idx in range(1,10):
      predict_data = loadmat(model_val_data[batch_idx+idx])
      #a01,a_gt0 = predict_data['echo_tmp'], predict_data['raster']
      a01 = predict_data['echo_tmp']
      
      if model.input_shape[-1]  >1:
          a0 = np.stack((a01,)*3,axis=-1)
          res0 = model.predict ( np.expand_dims(a0,axis=0))
      else:
          res0 = model.predict ( np.expand_dims(np.expand_dims(a01,axis=0),axis=3) ) 
          
      res0 = [ item.squeeze() for item in res0 ] # Multi_output model
      res0_final = np.sum(res0, axis=0)
      
      bin_thresh = np.percentile(res0_final, 70)      
      res0_final = np.where(res0>bin_thresh,1,0)
      
       # res0_final = sc_med_filt( sc_med_filt(res0_final.T,size=3).T, size= 3)      
      if decimated_model:
          res0_final1 = np.arange(1,417).reshape(416,1) * fix_final_prediction(res0,res0_final)
      else:
          res0_final1 = np.arange(1,416*4+1).reshape(416*4,1) * fix_final_prediction(res0,res0_final)
          
          
      # How correct is create_vec_layer??
      thresh = 5
      z = create_vec_layer(res0_final1,thresh); 
      
      z[z==0] = np.nan;
      
      b = (np.ones((7,1))/7).squeeze(); a = 1;          
      z_filtered =  filtfilt(b,a,z).astype('int32') #sc_med_filt(z,size=3)
      
      f, axarr = plt.subplots(1,4,figsize=(20,20))
    
      axarr[0].imshow(a01.squeeze(),cmap='gray_r')
      axarr[0].set_title( f'Echo {os.path.basename(model_val_data[batch_idx+idx])}') #.set_text
      
      axarr[1].imshow(res0_final.astype(bool).astype(int), cmap='viridis' )
      axarr[1].set_title('Prediction')        
      
      # axarr[2].plot(z.T) # gt
      # axarr[2].invert_yaxis()
      # axarr[2].set_title( f'Vec_layer({thresh})') #.set_text
      
      # axarr[3].imshow(a01.squeeze(),cmap='gray_r')          
      # axarr[3].plot(z.T) # gt
      # axarr[3].set_title( 'Overlaid prediction') #.set_textn') #.set_text



# Predict on the entire test data
test_pred = model.predict(x_test)
test_pred = np.argmax(test_pred,axis =3)

from sklearn.metrics import classification_report
print( classification_report( y_test.flatten(),test_pred.flatten(), labels=list(range(num_classes)), zero_division=1 ))


IoU = MeanIoU(num_classes = num_classes)
IoU.update_state(y_test.flatten(), test_pred.flatten() );
MIoU = IoU.result().numpy()
print(f'Mean IoU is {100*MIoU:.2f}%')

#roc_auc_score(y_test.flatten(), test_pred.flatten(),multi_class='OvR', labels=list(range(30)) )


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
    test_result = test_result.reshape((30,-1))
    
    return test_result
                  
            
layer_thickness = compute_layer_thickness(model, x_test)        
        
        
(np.mean(layer_thickness,axis =1)).round()














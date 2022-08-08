# -*- coding: utf-8 -*-
"""
Created 21st June,2022

@author: i368o351

"""

from cgi import test
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

from keras.metrics import MeanIoU
from sklearn.metrics import roc_auc_score

import glob
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,ReduceLROnPlateau
#import segmentation_models as sm
from datetime import datetime


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
    wandb.init( project="my-test-project", entity="ibksolar", name='deeplab'+time_stamp,config ={})
    config = wandb.config
else:
    config={}


try:
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
config['num_classes'] = 30
SEED = 42

config['img_channels'] = 3

AUTO = tf.data.experimental.AUTOTUNE


# =============================================================================
# Function for training data
def new_read_mat_train(filepath):
    def _read_mat(filepath):
        
        filepath = bytes.decode(filepath.numpy())      
        mat_file = loadmat(filepath)
        echo = tf.cast(mat_file['echo_tmp'], dtype=tf.float64)         
        layer = tf.cast(mat_file['semantic_seg'], dtype=tf.float64)          
        # layer = orig_mask = tf.image.resize(mat_file['semantic_seg'], (416,64))
        
        echo = tf.expand_dims(echo, axis=-1)
        
        if config['img_channels'] > 1:                                   
            echo = tf.image.grayscale_to_rgb(echo)
           
        
        # Data Augmentation        
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

                          
        #layer = tf.keras.utils.to_categorical(layer, config['num_classes'] )           
        layer = np.expand_dims(layer, axis=-1)                 
        shape0 = echo.shape #mat_file['echo_tmp'].shape 
        
        return echo,layer,np.asarray(shape0)
    
    output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double, tf.int64])
    shape = output[2]
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([416,64,3])
    
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
        layer = np.expand_dims(layer, axis=-1) 
        
        #layer = tf.keras.utils.to_categorical(layer, config['num_classes'] )
        shape0 = echo.shape #mat_file['echo_tmp'].shape        
        return echo,layer,np.asarray(shape0)     
    
    output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double, tf.int64])
    shape = output[2]
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([416,64,3])
    
    data1 = output[1]   
    data1.set_shape([416,64,1]) #,30   
    return data0,data1

train_ds = tf.data.Dataset.list_files(train_path,shuffle=True) #'*.mat'
train_ds = train_ds.map(new_read_mat_train,num_parallel_calls=8)
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

# Training params
#config={}
config['epochs'] = 700
config['learning_rate'] = 5e-4
config['num_classes']= 30



def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(), #glorot_normal,  #HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

input_shape = (416, 64, 3)
img_size_y, img_size_x = 416, 64

 
def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=image_size)
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor = model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    #x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    x = DilatedSpatialPyramidPooling(x)
    #x = layers.LayerNormalization(epsilon=1e-6)(x)

    input_a = layers.UpSampling2D(
        size=(img_size_y // 4 // x.shape[1], img_size_x // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)
    
    #input_b = layers.LayerNormalization(epsilon=1e-6)(input_b)

    x = layers.Concatenate(axis=-1)([input_a, input_b])  
    
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(img_size_y // x.shape[1], img_size_x // x.shape[2]),
        interpolation="bilinear",
    )(x)
    
    # x = layers.Conv2D(10*num_classes, kernel_size=(1, 1), padding="same", activation='relu')(x)
    # x = layers.Conv2D(5*num_classes, kernel_size=(1, 1), padding="same", activation='relu')(x)
    
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same",)(x) #, activation='softmax'
    return keras.Model(inputs=model_input, outputs=model_output)



model = DeeplabV3Plus(image_size=input_shape, num_classes=config['num_classes'])
#model.summary()

opt1 = keras.optimizers.Adam(learning_rate=config['learning_rate']) #,clipnorm=1, clipvalue=0.5
opt2 = keras.optimizers.SGD(learning_rate=config['learning_rate'],momentum = 0.9,clipnorm=0.5 )
opt3 = tfa.optimizers.AdamW(weight_decay = 0.001, learning_rate=config['learning_rate'])

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True) #keras.losses.CategoricalCrossentropy(),keras.losses.SparseCategoricalCrossentropy(), #from_logits = True

model.compile(
    optimizer= opt1,
    loss=loss,
    metrics=["accuracy"],
)

config['base_path'] = base_path                   
config['start_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
logz= f"{config['base_path']}//DeepLab//{config['start_time']}_logs/"
callbacks = [
   ModelCheckpoint(f"{config['base_path']}//DeepLab//DeepLab_Checkpoint{time_stamp}.h5", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=0.000005, verbose= 1),
    EarlyStopping(monitor="val_loss", patience=30, verbose=1), 
    TensorBoard(log_dir = logz,histogram_freq = 1,profile_batch = '1,70', embeddings_freq=50),
    WandbCallback()
]

history = model.fit(train_ds, validation_data=val_ds, epochs=config["epochs"], callbacks=callbacks)

plt.plot(history.history["loss"])
plt.title("Training Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["accuracy"])
plt.title("Training Accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["val_loss"])
plt.title("Validation Loss")
plt.ylabel("val_loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["val_accuracy"])
plt.title("Validation Accuracy")
plt.ylabel("val_accuracy")
plt.xlabel("epoch")
plt.show()



_,acc = model.evaluate(test_ds)
model.save(f"{config['base_path']}//Deeplab//DeepLab_acc_{acc:.2f}_{time_stamp}.h5")



model_val_data_path = os.path.join(base_path,'new_test\image\*.mat')
model_val_data = glob.glob(model_val_data_path)
batch_idx = random.randint(1,len(model_val_data)) # Pick any of the default batch

for idx in range(1,10):
  predict_data = loadmat(model_val_data[batch_idx+idx])
  a01,a_gt0 = predict_data['echo_tmp'], predict_data['semantic_seg']
  if config['img_channels']  >1:
      a0 = np.stack((a01,)*3,axis=-1)
      res0 = model.predict ( np.expand_dims(a0,axis=0))
  else:
      res0 = model.predict ( np.expand_dims(np.expand_dims(a0,axis=0),axis=3) ) 
      
  res0 = res0.squeeze()
  res0_final = np.argmax(res0,axis=2)
  
  pred0_final = sc_med_filt( sc_med_filt(res0_final,size=7).T, size=7, mode='nearest').T

  f, axarr = plt.subplots(1,4,figsize=(20,20))

  axarr[0].imshow(a01.squeeze(),cmap='gray_r')
  axarr[0].set_title( f'Echo {os.path.basename(model_val_data[batch_idx])}') #.set_text

  axarr[1].imshow(res0_final, cmap='viridis' )
  axarr[1].set_title('Prediction')
  
  axarr[2].imshow(pred0_final, cmap='viridis') 
  axarr[2].set_title('Filtered Prediction')
  
  axarr[3].imshow(a_gt0.squeeze(),cmap='viridis') # gt
  axarr[3].set_title( f'Ground truth {os.path.basename(model_val_data[batch_idx])}') #.set_text


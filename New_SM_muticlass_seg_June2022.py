# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 12:29:36 2021

Segmentation_models
%env SM_FRAMEWORK=tf.keras
@author: i368o351
"""

%env SM_FRAMEWORK=tf.keras

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
import random
from scipy.io import loadmat
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,ReduceLROnPlateau
from matplotlib import cm,colors


import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()

import tensorflow_addons as tfa

import glob

from datetime import datetime

# Instantiations to train model

backbone1 = 'resnet34'
backbone2 = 'resnet50'


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

use_wandb = False
if use_wandb:
    ## WandB config
    import wandb
    from wandb.keras import WandbCallback
    wandb.init( project="my-test-project", entity="ibksolar", name='NewSM_MultiClass_Seg'+time_stamp,config ={})
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
base_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Train_data'  # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'
train_path = os.path.join(base_path,'train_data\*.mat')
val_path = os.path.join(base_path,'val_data\*.mat')
test_path = os.path.join(base_path,'test_data\*.mat')   

# Create tf.data.Dataset
config['batch_size'] = 16
config['num_classes'] = 30 # This needs to be UP here for dataset creation

config['num_epochs'] = 500

SEED = 42
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
        layer = tf.expand_dims(layer, axis=-1)
        
        
        echo = tf.image.grayscale_to_rgb(echo)
        echo = tf.cast(echo, tf.float64) / 255.0        
        
        
        # Data Augmentation        
        if  tf.random.uniform(())> 0.1:
            aug_type = tf.random.uniform((1,1),minval=1, maxval=7,dtype=tf.int64).numpy()
            
            if aug_type == 1:
                #echo = tf.experimental.numpy.fliplr(echo)
                #layer = tf.experimental.numpy.fliplr(layer)
                
                echo = tf.image.flip_left_right(echo)
                layer = tf.image.flip_left_right(layer)
            
            elif aug_type == 2: # Constant offset
                echo = echo - 0.3
            
            elif aug_type == 3: # Random saturation
                #echo = echo - tf.random.normal(shape=(416,64),stddev=0.5,dtype=tf.float64)
                echo = tf.image.adjust_saturation(echo, 0.2)
                
            elif aug_type == 4: # Random brightness
                echo = echo - tf.image.random_brightness(echo,0.2)
            
            elif aug_type == 5: # Random contrast
                echo = echo - tf.image.random_contrast(echo,0.2,0.5)
                
            elif aug_type == 4: # Random rotation            
                rot_factor = tf.cast(tf.random.uniform(shape=[], maxval=12, dtype=tf.int32), tf.float32)
                angle = np.pi/12*rot_factor
                echo = tfa.image.rotate(echo,angle)
                layer = tfa.image.rotate(layer,angle)
            
            else: #aug_type == 4:
                # echo = tf.experimental.numpy.flipud(echo)
                # layer = tf.experimental.numpy.flipud(layer)

                echo = tf.image.flip_up_down(echo)
                layer = tf.image.flip_up_down(layer)                            
                   
        
        layer = tf.keras.utils.to_categorical(layer, config['num_classes'] )
        shape0 = echo.shape #mat_file['echo_tmp'].shape        
        return echo,layer,np.asarray(shape0)
    
    output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double, tf.int64])
    shape = output[2]
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([416,64,3])
    
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
        echo = tf.image.grayscale_to_rgb(echo)
        echo = tf.cast(echo, tf.float64) / 255.0
        
        layer = tf.cast(mat_file['semantic_seg'], dtype=tf.float64)      

        layer = tf.keras.utils.to_categorical(layer, config['num_classes'] )
        shape0 = echo.shape #mat_file['echo_tmp'].shape        
        return echo,layer,np.asarray(shape0)     
    
    output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double, tf.int64])
    shape = output[2]
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([416,64,3])
    
    data1 = output[1]   
    data1.set_shape([416,64,30 ]) #,30   
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


# Callbacks
config['base_path'] = base_path #os.path.abspath(r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\PulsedTrainTest').replace(os.sep,'/')
config['start_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
logz= f"{config['base_path']}//echo_vit//{config['start_time']}_logs/"
callbacks = [
   ModelCheckpoint(f"{config['base_path']}//SM_multiclass//SM_multiclass_Checkpoint{time_stamp}.h5", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=0.00005, verbose= 1),
    EarlyStopping(monitor="val_loss", patience=30, verbose=1), 
    #TensorBoard(log_dir = logz,histogram_freq = 1,profile_batch = '1,70', embeddings_freq=50),
    #WandbCallback()
]


# define model
ResNet50_model = sm.Unet(backbone2, encoder_weights='imagenet', classes = config['num_classes'], activation = 'softmax')
ResNet50_model.compile('NAdam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy', sm.metrics.iou_score])
#print(ResNet50_model.summary())

#Fit the model
#history = model.fit(my_generator, validation_data=validation_datagen, steps_per_epoch=len(X_train) // 16, validation_steps=len(X_train) // 16, epochs=100)
#history = model.fit(my_generator, validation_data=validation_datagen, steps_per_epoch=5, validation_steps=5, epochs=5, callbacks=[ WandbCallback() ])

time_stamp = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
print(f'Training started {time_stamp}')


history = ResNet50_model.fit(train_ds, epochs= config['num_epochs'], validation_data= val_ds, callbacks = callbacks)


time_stamp = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
print(f'Training ended {time_stamp}')



_,acc,iou = ResNet50_model.evaluate(test_ds)

model_save_path = f'{base_path}/SM_multiclass_segmentation/ResNet50_Acc_{acc*100: .2f}_Epoch{config["num_epochs"]}_{time_stamp}.h5'

ResNet50_model.save(model_save_path)

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['iou_score']
#acc = history.history['accuracy']
val_acc = history.history['val_iou_score']
#val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training IOU')
plt.plot(epochs, val_acc, 'r', label='Validation IOU')
plt.title('Training and validation IOU')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.legend()
plt.show()


custom_cm = cm.Blues(np.linspace(0,1,30))
custom_cm = colors.ListedColormap(custom_cm[10:,:-1])

# import random

## Visualize result of model prediction for "unseen" echogram during training
model_val_data_path = os.path.join(base_path,'test_data\*.mat')
model_val_data = glob.glob(model_val_data_path)

batch_idx = random.randint(1,len(model_val_data)) # Pick any of the default batch

for idx in range(5):
  predict_data = loadmat(model_val_data[batch_idx+idx])
  a0,a_gt0 = predict_data['echo_tmp'], predict_data['semantic_seg']
  a0 = np.stack((a0,)*3, axis=-1)
    
  # a0 = a[idx]
  # a_gt0 = a_gt[idx]
  # ( a0.shape, a_gt0.shape )

  res0 = ResNet50_model.predict ( np.expand_dims(a0,axis=0) ) 
  res0 = res0.squeeze()
  res0_final = np.argmax(res0,axis=2)


  f, axarr = plt.subplots(1,3,figsize=(20,20))

  axarr[0].imshow(a0.squeeze(),cmap='gray_r')
  axarr[0].set_title( f'Echo {os.path.basename(model_val_data[batch_idx+idx])}') #.set_text

  axarr[1].imshow(a_gt0.squeeze(),cmap=custom_cm) # gt
  axarr[1].set_title( 'Ground truth') #.set_text

  axarr[2].imshow(res0_final, cmap=custom_cm )
  axarr[2].set_title('Prediction')








# if 0:    
#     #IOU
#     x_test2 =preprocess_input1(x_test)
#     y_pred=model.predict(x_test2)
#     y_pred_thresholded = y_pred > 0.5
    
#     intersection = np.logical_and(y_test, y_pred_thresholded)
#     union = np.logical_or(y_test, y_pred_thresholded)
#     iou_score = np.sum(intersection) / np.sum(union)
#     print("IoU socre is: ", iou_score)
    
#     test_img_number = random.randint(0, len(x_test)-1)
#     test_img = x_test[test_img_number]
#     test_img_input=np.expand_dims(test_img, 0)
#     ground_truth=y_test[test_img_number]
#     prediction = model.predict(test_img_input)
#     prediction = prediction[0,:,:,0]
    
#     plt.figure(figsize=(16, 8))
#     plt.subplot(231)
#     plt.title('Testing Image')
#     plt.imshow(test_img[:,:,0], cmap='gray')
#     plt.subplot(232)
#     plt.title('Testing Label')
#     plt.imshow(ground_truth[:,:,0], cmap='gray')
#     plt.subplot(233)
#     plt.title('Prediction on test image')
#     plt.imshow(prediction, cmap='gray')








































































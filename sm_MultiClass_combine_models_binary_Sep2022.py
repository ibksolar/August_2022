# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 18:40:01 2021

@author: i368o351
"""

# %env SM_FRAMEWORK=tf.keras

import tensorflow as tf
import os
from tensorflow import keras
import numpy as np
import random
from tensorflow.keras import backend as K
from scipy.io import loadmat,savemat
from matplotlib import pyplot as plt
from matplotlib import cm
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,ReduceLROnPlateau
from scipy.ndimage import median_filter as sc_med_filt


import glob

from datetime import datetime
# from PSPNet_implementation import last_conv_module

# !pip install segmentation-models 
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()

from tensorflow.keras import mixed_precision
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
    wandb.init( project="my-test-project", entity="ibksolar", name='SM_multi_multiclass'+time_stamp,config ={})
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
base_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data'
# base_path = r'K:\Users\cresis\Desktop\Fall_2021\all_block_data\Attention_Train_data\new_trainJuly'   # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'
train_path = os.path.join(base_path,'train_data\*.mat')
val_path = os.path.join(base_path,'val_data\*.mat')
test_path = os.path.join(base_path,'test_data\*.mat')   

# Create tf.data.Dataset
config['batch_size'] = 4


# Training params
config['img_y'] = IMG_HEIGHT = 416*4
config['img_x'] = IMG_WIDTH = 64*4

config['img_channels'] = IMG_CHANNELS = 3
config['weight_decay'] = 0.0001

config['num_classes'] = 1 #30 # This needs to be UP here for dataset creation
config['epochs'] = 500
config['learning_rate'] = 1e-3
config['base_path'] = base_path
SEED = 42

AUTO = tf.data.experimental.AUTOTUNE


# =============================================================================
# Function for creating train, test, validation data
def old_read_mat_train(filepath):
    def _read_mat(filepath):
        
        filepath = bytes.decode(filepath.numpy())      
        mat_file = loadmat(filepath)
        echo = tf.cast(mat_file['echo_tmp'], dtype=tf.float64)
        layer = tf.cast(mat_file['semantic_seg2'], dtype=tf.float64)
        
        # Data Augmentation
        
        if tf.random.uniform(())> 0.5:
            aug_type = tf.random.uniform((1,1),minval=1, maxval=4,dtype=tf.int64).numpy()
            
            if aug_type == 1:
                echo = tf.experimental.numpy.fliplr(echo)
                layer = tf.experimental.numpy.fliplr(layer)
            
            elif aug_type == 2: # Constant offset
                echo = echo - 0.3
            
            elif aug_type == 3: # Random noise
                echo = echo - tf.random.normal(shape=(416,64),stddev=0.5,dtype=tf.float64)
                
            else: #aug_type == 4:
                echo = tf.experimental.numpy.flipud(echo)
                layer = tf.experimental.numpy.flipud(layer)                            

        echo = tf.expand_dims(echo, axis=-1)
        echo = tf.image.grayscale_to_rgb(echo)                   
        
        layer = tf.keras.utils.to_categorical(layer, config['num_classes'] )
        shape0 = echo.shape #mat_file['echo_tmp']
        
        return echo,layer,np.asarray(shape0)
    output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double, tf.int64])
    shape = output[2]
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([416,64,3])
    
    data1 = output[1]   
    data1.set_shape([416,64,30])#    
    return data0,data1

# =============================================================================
## Function for test and validation dataset    
def old_read_mat(filepath):
    def _read_mat(filepath):
        
        filepath = bytes.decode(filepath.numpy())      
        mat_file = loadmat(filepath)
        echo = tf.cast(mat_file['echo_tmp'], dtype=tf.float64)
        
        echo = tf.expand_dims(echo, axis=-1)
        echo = tf.image.grayscale_to_rgb(echo)

        
        layer = tf.cast(mat_file['semantic_seg2'], dtype=tf.float64)      

        layer = tf.keras.utils.to_categorical(layer, config['num_classes'] )
        shape0 = echo.shape #mat_file['echo_tmp'].shape        
        return echo,layer,np.asarray(shape0)     
    
    output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double, tf.int64])
    shape = output[2]
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([416,64,3])
    
    data1 = output[1]   
    data1.set_shape([416,64,30]) #  ,30
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
        
        layer = tf.cast(mat_file['raster'], dtype=tf.float64)
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
    data0.set_shape([config['img_y'],config['img_x'],config['img_channels']])

    
    data1 = output[1]   
    data1.set_shape([config['img_y'],config['img_x'],1])
    # data1.set_shape([1664,256,1 ])#,30,config['num_classes']    
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
        
        layer = tf.cast(mat_file['raster'], dtype=tf.float64)      
        
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


# Instantiations to train model

backbone1 = 'resnet34'
backbone2 = 'resnet50'

resnet34_preprocess = sm.get_preprocessing(backbone1)
resnet50_preprocess = sm.get_preprocessing(backbone2)

x_train34 = resnet34_preprocess(train_ds)
x_val34 = resnet34_preprocess(val_ds)
x_test34 = resnet34_preprocess(test_ds)

x_train50 = resnet50_preprocess(train_ds)
x_val50 = resnet50_preprocess(val_ds)
x_test50 = resnet50_preprocess(test_ds)


# Losses
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss(gamma=5.0)
total_loss = dice_loss + (2*focal_loss)


config['epochs'] = 250 #700
config['learning_rate'] = 1e-3


# Create Callbacks
config['base_path'] = base_path
config['start_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
logz= f"{config['base_path']}/SM_combine_binary//{config['start_time']}_logs/"

# loaded_model = tf.keras.models.load_model('/content/gdrive/MyDrive/echo_cnn_in_out_GOOD_layers/SM_UNet_ResNet50_segment_96percent_210621',custom_objects={'binary_crossentropy_plus_jaccard_loss': sm.losses.bce_jaccard_loss, 'iou_score':sm.metrics.iou_score})


# ================================================================================
    # Train different models
# ================================================================================

train_models = 1

input_shape = (config['img_y'], config['img_x'], config['img_channels'])

if train_models:
    
    # Different models
    sm_UnetResNet34_model = sm.Unet(backbone_name= backbone1, encoder_weights='imagenet', input_shape = input_shape, classes=config['num_classes'], encoder_freeze=True, activation='sigmoid')
    sm_UnetResNet50_model = sm.Unet(backbone_name= backbone2, encoder_weights='imagenet', input_shape = input_shape, classes=config['num_classes'],encoder_freeze=True, activation='sigmoid')
    sm_FPNmodel = sm.FPN(backbone_name='resnet50', encoder_weights='imagenet', input_shape = input_shape, classes=config['num_classes'], encoder_freeze=True, activation='sigmoid')
    
    # Model 1 
    
    callbacks = [
       ModelCheckpoint(f"{config['base_path']}//SM_combine_binary//{backbone1}_Checkpoint{time_stamp}.h5", save_best_only=True, monitor="val_loss"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.25, patience=10, min_lr=0.000005, verbose= 1),
        EarlyStopping(monitor="val_loss", patience=30, verbose=1), 
        TensorBoard(log_dir = logz,histogram_freq = 1,profile_batch = '1,70', embeddings_freq=50),
        WandbCallback()
    ]
    
    sm_UnetResNet34_model.compile(
        'Adam',
        loss = tf.keras.losses.CategoricalCrossentropy(), #total_loss
        metrics=['accuracy',sm.metrics.iou_score,], #sm.metrics.iou_score,
    )
    #sm_UnetResNet34_model_history = sm_UnetResNet34_model.fit(x_train34, epochs=config['epochs'], validation_data= x_val34, callbacks= callbacks )
    sm_UnetResNet34_model_history = sm_UnetResNet34_model.fit(train_ds, epochs=config['epochs'], validation_data= val_ds, callbacks= callbacks )
    
    #sm_UnetResNet34_model.save('sm_UnetResNet34_model')
    # ================================================================================
    
    # Model 2
    # ================================================================================
    callbacks2 = [
       ModelCheckpoint(f"{config['base_path']}//SM_combine_binary//{backbone2}_Checkpoint{time_stamp}.h5", save_best_only=True, monitor="val_loss"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=0.000005, verbose= 1),
        EarlyStopping(monitor="val_loss", patience=30, verbose=1), 
        TensorBoard(log_dir = logz,histogram_freq = 1,profile_batch = '1,70', embeddings_freq=50),
        WandbCallback()
    ]
    sm_UnetResNet50_model.compile(
        'Adam',
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy',sm.metrics.iou_score,],
    )
    sm_UnetResNet50_model_history =  sm_UnetResNet50_model.fit(train_ds, epochs=config['epochs'], validation_data= test_ds, callbacks= callbacks2 )
    
    
    #sm_UnetResNet50_model.save('sm_UnetResNet50_model')
    # ================================================================================
    
    # Model 3
    
    callbacks3 = [
       ModelCheckpoint(f"{config['base_path']}//SM_combine_binary//FPN_Checkpoint{time_stamp}.h5", save_best_only=True, monitor="val_loss"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=0.000005, verbose= 1),
        EarlyStopping(monitor="val_loss", patience=30, verbose=1), 
        TensorBoard(log_dir = logz,histogram_freq = 1,profile_batch = '1,70', embeddings_freq=50),
        WandbCallback()
    ]
    
    sm_FPNmodel.compile(
        'Adam',
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy',sm.metrics.iou_score ],
    )
    
    sm_FPNmodel_history =  sm_FPNmodel.fit(train_ds, epochs=config['epochs'], validation_data= val_ds, callbacks= callbacks3  )
    
    #sm_FPNmodel.save('sm_FPNmodel')
    
        
    time_stamp = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
    
    _,iou_score34,sm_UnetResNet34_model_acc = sm_UnetResNet34_model.evaluate(test_ds)
    _,iou_score50,sm_UnetResNet50_model_acc = sm_UnetResNet50_model.evaluate(test_ds)
    _,iou_scoreFPN,sm_FPNmodel_acc = sm_FPNmodel.evaluate(test_ds)
    
    #sm_UnetResNet34_model
    model_save_path = f'{base_path}/SM_combine_binary/{backbone1}_{sm_UnetResNet34_model_acc*100: .2f}_Epochs_{config["epochs"]}_{time_stamp}.h5'
    sm_UnetResNet34_model.save(model_save_path)
    
    model_save_path = f'{base_path}/SM_combine_binary/{backbone2}_model_Acc{sm_UnetResNet50_model_acc*100: .2f}_Epochs_{config["epochs"]}_{time_stamp}.h5'
    sm_UnetResNet50_model.save(model_save_path)
    
    model_save_path = f'{base_path}/SM_combine_binary/FPN_model_acc_{sm_FPNmodel_acc*100: .2f}_Epochs_{config["epochs"]}_{time_stamp}.h5'
    sm_FPNmodel.save(model_save_path)
    
    loaded_FPN = sm_FPNmodel
    loaded_Res34 = sm_UnetResNet34_model
    loaded_Res50 = sm_UnetResNet50_model
    # Attempt loading model to be sure
#    loaded_model = tf.keras.models.load_model(model_save_path
#                                    ,custom_objects={'CCTTokenizer':CCTTokenizer,'StochasticDepth':StochasticDepth})



else:
    # Load trained models ( not training)
    loaded_FPN = tf.keras.models.load_model('sm_FPNmodel',custom_objects={'dice_loss': sm.losses.DiceLoss(),'focal_loss':sm.losses.BinaryFocalLoss(gamma=5.0), 'iou_score':sm.metrics.iou_score, 'dice_loss_plus_2binary_focal_loss':sm.losses.DiceLoss()+2*sm.losses.BinaryFocalLoss(gamma=5.0) })
    loaded_Res34 = tf.keras.models.load_model('sm_UnetResNet34_model',custom_objects={'dice_loss': sm.losses.DiceLoss(),'focal_loss':sm.losses.BinaryFocalLoss(gamma=5.0), 'iou_score':sm.metrics.iou_score, 'dice_loss_plus_2binary_focal_loss':sm.losses.DiceLoss()+2*sm.losses.BinaryFocalLoss(gamma=5.0) })
    loaded_Res50 = tf.keras.models.load_model('sm_UnetResNet50_model',custom_objects={'dice_loss': sm.losses.DiceLoss(),'focal_loss':sm.losses.BinaryFocalLoss(gamma=5.0), 'iou_score':sm.metrics.iou_score, 'dice_loss_plus_2binary_focal_loss':sm.losses.DiceLoss()+2*sm.losses.BinaryFocalLoss(gamma=5.0) })

# Train further (12th July,2021)
train_further = 0

if train_further:
    epochs = 10
    loaded_FPN.fit(x_train50, epochs=epochs, validation_data= x_val50 )
    loaded_Res34.fit(x_train34, epochs=epochs, validation_data= x_val34 )
    loaded_Res50.fit(x_train50, epochs=epochs, validation_data= x_val50 )
    


# ====================================================================================
    # Visualize result for different models e.g Res50
# ====================================================================================


model_val_data_path = os.path.join(base_path,'test\*.mat')
model_val_data = glob.glob(model_val_data_path)


# # Visualize result of model prediction for "unseen" echogram during training
model = sm_FPNmodel #sm_UnetResNet50_model #sm_UnetResNet50_model #sm_FPNmodel <= CHANGE HERE based on model to test


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
  
  if config['num_classes']> 1:
      res0_final = np.argmax(res0,axis=2)
  else:
      res0_final = np.where(res0>0.4,1,0)
      
  res_filtered = sc_med_filt( sc_med_filt(res0_final,size=7).T, size=7, mode='nearest').T 

  f, axarr = plt.subplots(1,4,figsize=(20,20))

  axarr[0].imshow(a01.squeeze(),cmap='gray_r')
  axarr[0].set_title( f'Echo {os.path.basename(model_val_data[batch_idx])}') #.set_text
  
  axarr[1].imshow(res0_final, cmap='viridis' )
  axarr[1].set_title('Prediction')
  
  axarr[2].imshow(res_filtered, cmap='viridis' )
  axarr[2].set_title('Filtered_prediction')

  axarr[3].imshow(a_gt0.squeeze(),cmap='viridis') # gt
  axarr[3].set_title( f'Ground truth {os.path.basename(model_val_data[batch_idx])}') #.set_text

  
# =============================================================================
        # Evaluate Mean IoU for each iteration
# =============================================================================
  
  IoU_FPN = MeanIoU(num_classes = config['num_classes'])
  IoU_Res34 = MeanIoU(num_classes = config['num_classes'])
  IoU_Res50 = MeanIoU(num_classes = config['num_classes'])
  
  IoU_FPN.update_state(a_gt0,res0_final );
  print(f' FPN IoU is {IoU_FPN.result().numpy() }')
  
  IoU_Res34.update_state(a_gt0,res34_final );
  print(f' Res34 IoU is {IoU_Res34.result().numpy() }')
  
  IoU_Res50.update_state(a_gt0,res50_final );
  print(f' Res50 IoU is {IoU_Res34.result().numpy() }')
  
  preds = np.array([res0, res34, res50] )
  weights = [0.2,0.5,0.3]
  weighted_preds = np.tensordot(preds,weights,axes=( (0),(0) ))
  ensenmble_pred = np.argmax(weighted_preds, axis = 2)
  
# =============================================================================
        # Plots
# =============================================================================

  f, axarr = plt.subplots(1,6,figsize=(15,15))
  
  # plot image
  axarr[0].imshow(a0.squeeze(),cmap='gray_r')
  axarr[0].set_title( f'Echo {curr_path}') #.set_text

  # Plot ground truth  
  axarr[1].imshow(gt,cmap=cm.get_cmap('viridis', 30)) # gt
  axarr[1].set_title( 'GT') #.set_text
  
  # axarr[2].imshow( np.argmax(a_gt0,axis=2) ,cmap=cm.get_cmap('viridis', 30)) # gt
  # axarr[2].set_title( f'GT2(Batch) {curr_path}') #.set_text
  
  axarr[2].imshow( ensenmble_pred ,cmap=cm.get_cmap('viridis', 30)) # gt
  axarr[2].set_title( 'Ensemble') #.set_text

  # Plot predictions
  axarr[3].imshow(res0_final, cmap=cm.get_cmap('viridis', 30) )
  axarr[3].set_title('FPN_Prediction')
  
  axarr[4].imshow(res34_final, cmap=cm.get_cmap('viridis', 30) )
  axarr[4].set_title('Res34_Prediction')

  axarr[5].imshow(res50_final, cmap=cm.get_cmap('viridis', 30) )
  axarr[5].set_title('Res50_Prediction')
  
# ====================================================================================
# ====================================================================================

  
# Combine models
## Combine for test data

FPN_pred = loaded_FPN.predict ( x_test50 )
Res34_pred = loaded_Res34.predict ( x_test34 )
Res50_pred = loaded_Res50.predict ( x_test50 )

preds = np.array([FPN_pred,Res34_pred,Res50_pred] )
weights = [0.2,0.5,0.3]

weighted_preds = np.tensordot(preds,weights,axes=( (0),(0) ))
ensenmble_preds = np.argmax(weighted_preds, axis = 3)


 # Find best combination weights to max MeanIoU
# =============================================================================
c = np.concatenate( [ test_gen[idx][1] for idx in range(len(test_gen)) ] )
c2 = np.argmax(c,axis = 3)

resIoU =[]
for w1 in np.arange(0.0, 1.0, 0.1):
    for w2 in np.arange(0.0, 1.0, 0.1):
        w3 = 1 - (w1+w2)       
       
        opt_weights = np.array( [w1,w2,w3] )
        
        weighted_preds_loop = np.tensordot(preds,opt_weights,axes=( (0),(0) ))
        ensenmble_preds_loop = np.argmax(weighted_preds_loop, axis = 3)
        
        combineIoU = MeanIoU(num_classes = num_classes);
        combineIoU.update_state(c2,ensenmble_preds_loop)
        
        resIoU.append( (w1,w2,w3, combineIoU.result().numpy()) )
        
        print(f' W1:{w1}, W2:{w2}, W3:{w3}, MeanIoU: { combineIoU.result().numpy() } ')

# best_weight = sorted(resIoU,key=lambda x: x[3],reverse=True )[0][0:-1]            
best_weight = [0.2,0.5,0.3]            
            

# Save Model predictions to MATLAB
save_flag = 0
if save_flag:
    result_to_save = {}
    
    result_to_save['FPN'] = FPN_pred
    result_to_save['Res34'] = Res34_pred
    result_to_save['Res50'] = Res50_pred    
    fn = 'ML_multi_class_prediction.mat'
    savemat(fn,result_to_save)



# Save each model and ensemble prediction result
if 0:    
    save_dict = {}
    save_dict['FPN_test_pred'] = FPN_pred
    save_dict['Res34_test_pred'] = Res34_pred
    save_dict['Res50_test_pred'] = Res50_pred
    save_dict['Combined_test_pred'] = weighted_preds
    save_dict['test_paths'] = test_input_img_paths
    
    savemat('Combined_model_pred.mat',save_dict) # Saved into Colab models directory













# -*- coding: utf-8 -*-
"""
Created 4th June,2022
Modifying AttentionUNet for EchoViT paper

TODO:
    Only Binary case works but Multi_class isnt working. (Need to fix that)
    using attention gate on both spatial and channel dimensions
    
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
sm.set_framework('tf.keras')
sm.framework()

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

wandb.init( project="my-test-project", entity="ibksolar", name='AttnUNet'+time_stamp,config ={})
config = wandb.config


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
config['num_classes'] = 30 # This needs to be UP here for dataset creation
config['dropout_rate'] = 0.5
config['learning_rate'] = 1e-3
config['epochs'] = 500

config['img_channels'] = 3; # Simulate either grayscale or rgb
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
    data0.set_shape([416,64,config['img_channels']])
    
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


# Network structure
filter_num = 128 # number of basic filters for the first layer
filter_size = (17,13) # shape/size of the convolutional filter

up_samp_size = 2 # size of upsampling filters
SE_RATIO = 2   # reduction ratio of SE block

# Input data
# Hyperparameters and constants

image_size_y = 416 #224, 416
image_size_x = 64
img_channels = 3  # 1-grayscale, 3-RGB scale

img_size = (image_size_y,image_size_x) # :( Almost redundant
input_shape = (image_size_y, image_size_x, img_channels) #, img_channels)



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


ALPHA = 0.8
GAMMA = 2
def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):    
    
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
    
    return focal_loss
##==============================================================================##


# # Losses
# dice_loss = sm.losses.DiceLoss()
# SM_focal_loss = sm.losses.BinaryFocalLoss(gamma=2.0)
# total_loss = dice_loss + (2*SM_focal_loss)



# Network Functions    

def expend_as(tensor, rep):
     return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)


def double_conv_layer(x, filter_size, size, dropout, batch_norm=False):
    '''
    construction of a double convolutional layer using
    SAME padding
    RELU nonlinear activation function
    :param x: input
    :param filter_size: size of convolutional filter
    :param size: number of filters
    :param dropout: FLAG & RATE of dropout.
            if < 0 dropout cancelled, if > 0 set as the rate
    :param batch_norm: flag of if batch_norm used,
            if True batch normalization
    :return: output of a double convolutional layer
    '''
    filter_size1,filter_size2 = filter_size
    
    axis = 3
    conv = layers.Conv2D(size, (filter_size1, filter_size2), padding='same')(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=axis)(conv)
    conv = layers.Activation('relu')(conv)
    conv = layers.Conv2D(size, (filter_size1, filter_size2), padding='same')(conv) # Sreeni removed this
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=axis)(conv)
    conv = layers.Activation('relu')(conv)
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)
    
    # Sr**ni did not add this part ( he returned conv from here (Error - this was another fxn))
    shortcut = layers.Conv2D(size, kernel_size=(1, 1), padding='same')(x)
    if batch_norm is True:
        shortcut = layers.BatchNormalization(axis=axis)(shortcut)

    res_path = layers.add([shortcut, conv])
    res_path = layers.Activation('relu')(res_path)
    return res_path

def SE_block(x, out_dim, ratio, name, batch_norm=False):
    """
    self attention squeeze-excitation block, attention mechanism on channel dimension
    :param x: input feature map
    :return: attention weighted on channel dimension feature map
    """
    # Squeeze: global average pooling
    x_s = layers.GlobalAveragePooling2D(data_format=None)(x)
    # Excitation: bottom-up top-down FCs
    if batch_norm:
        x_s = layers.BatchNormalization()(x_s)
    x_e = layers.Dense(units=out_dim//ratio)(x_s)
    x_e = layers.Activation('relu')(x_e)
    if batch_norm:
        x_e = layers.BatchNormalization()(x_e)
    x_e = layers.Dense(units=out_dim)(x_e)
    x_e = layers.Activation('sigmoid')(x_e)
    x_e = layers.Reshape((1, 1, out_dim), name=name+'channel_weight')(x_e)
    result = layers.multiply([x, x_e])
    return result



def gating_signal(input, out_size, batch_norm=False):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :param input:   down-dim feature map
    :param out_size:output channel number
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    x = layers.Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def attention_block(x, gating, inter_shape, name):
    """
    self gated attention, attention mechanism on spatial dimension
    :param x: input feature map
    :param gating: gate signal, feature map from the lower layer
    :param inter_shape: intermedium channel numer
    :param name: name of attention layer, for output
    :return: attention weighted on spatial dimension feature map
    """

    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_g)  # 16
    # upsample_g = layers.UpSampling2D(size=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
    #                                  data_format="channels_last")(phi_g)

    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]),
                                       name=name+'_weight')(sigmoid_xg)  # 32

    upsample_psi = expend_as(upsample_psi, shape_x[3])


    y = layers.multiply([upsample_psi, x])

    result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = layers.BatchNormalization()(result)
    return result_bn


def Attention_ResUNet_PA(input_shape,num_classes= config['num_classes'] ,dropout_rate=0.0, batch_norm=True):
    '''
    Residual UNet construction, with attention gate
    convolution: 3*3 SAME padding
    pooling: 2*2 VALID padding
    upsampling: 3*3 VALID padding
    final convolution: 1*1
    :param dropout_rate: FLAG & RATE of dropout.
            if < 0 dropout cancelled, if > 0 set as the rate
    :param batch_norm: flag of if batch_norm used,
            if True batch normalization
    :return: model
    '''
    # input data
    # dimension of the image depth
    inputs = layers.Input(shape = input_shape, dtype=tf.float64, name = "Main_input")    
    # inputs = tf.expand_dims(inputs, axis = -1)
    axis = 3
    
    filter_size_mtx = [ (17,13), (7,5), (3,3) ]

    # Downsampling layers
    # DownRes 1, double residual convolution + pooling
    conv_128 = double_conv_layer(inputs, filter_size_mtx[0], filter_num, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_64 = double_conv_layer(pool_64, filter_size_mtx[0], 2*filter_num, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = double_conv_layer(pool_32, filter_size_mtx[1], 4*filter_num, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = double_conv_layer(pool_16, filter_size_mtx[2], 8*filter_num, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = double_conv_layer(pool_8, filter_size_mtx[2], 16*filter_num, dropout_rate, batch_norm)

    # Upsampling layers

    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    # channel attention block
    #se_conv_16 = SE_block(conv_16, out_dim=8*filter_num, ratio=SE_RATIO, name='att_16')
    # spatial attention block
    gating_16 = gating_signal(conv_8, 8*filter_num, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 8*filter_num, name='att_16')
    # attention re-weight & concatenate
    up_16 = layers.UpSampling2D(size=(up_samp_size, up_samp_size), data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=axis)
    up_conv_16 = double_conv_layer(up_16, filter_size_mtx[2], 8*filter_num, dropout_rate, batch_norm)

    # UpRes 7
    # channel attention block
    #se_conv_32 = SE_block(conv_32, out_dim=4*filter_num, ratio=SE_RATIO, name='att_32')
    # spatial attention block
    gating_32 = gating_signal(up_conv_16, 4*filter_num, batch_norm) #up_conv_16
    att_32 = attention_block(conv_32, gating_32, 4*filter_num, name='att_32')  #se_conv_32
    # attention re-weight & concatenate
    up_32 = layers.UpSampling2D(size=(up_samp_size, up_samp_size), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=axis)
    up_conv_32 = double_conv_layer(up_32, filter_size_mtx[2], 4*filter_num, dropout_rate, batch_norm)

    # UpRes 8
    # channel attention block
    #se_conv_64 = SE_block(conv_64, out_dim=2*filter_num, ratio=SE_RATIO, name='att_64')
    # spatial attention block
    gating_64 = gating_signal(up_conv_32, 2*filter_num, batch_norm)
    att_64 = attention_block(conv_64, gating_64, 2*filter_num, name='att_64') #se_conv_64
    # attention re-weight & concatenate
    up_64 = layers.UpSampling2D(size=(up_samp_size, up_samp_size), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, att_64], axis=axis)
    up_conv_64 = double_conv_layer(up_64, filter_size_mtx[2], 2*filter_num, dropout_rate, batch_norm)

    # UpRes 9
    # channel attention block
    #se_conv_128 = SE_block(conv_128, out_dim=filter_num, ratio=SE_RATIO, name='att_128')
    # spatial attention block
    gating_128 = gating_signal(up_conv_64, filter_num, batch_norm)
    # attention re-weight & concatenate
    att_128 = attention_block(conv_128, gating_128, filter_num, name='att_128') #se_conv_128
    up_128 = layers.UpSampling2D(size=(up_samp_size, up_samp_size), data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, att_128], axis=axis)
    up_conv_128 = double_conv_layer(up_128, filter_size, filter_num, dropout_rate, batch_norm)
    
    # up_conv_128 = tf.reduce_sum(up_conv_128,axis=-1)
    up_conv_128 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up_conv_128)
    up_conv_128 = tf.keras.layers.Dropout(0.5)(up_conv_128)
    #outputs = tf.keras.layers.Conv2D(config['config['num_classes'] '] , (1, 1), activation='sigmoid')(up_conv_128) #sigmoid, softmax
    
    # 1*1 convolutional layers
    # valid padding
    # batch normalization
    # sigmoid nonlinear activation

    if config['num_classes'] >1:
        conv_final = layers.Conv2D(config['num_classes'], (1,1), padding="same",activation="softmax",  dtype = tf.float64) (up_conv_128)

    else:
        conv_final = tf.keras.layers.Conv2D(config['num_classes'] , (1, 1), activation='sigmoid')(up_conv_128) #relu, sigmoid, softmax

    # Model integration
    model = keras.Model(inputs, conv_final, name="AttentionSEResUNet")
    return model

# Instantiate a model
model = Attention_ResUNet_PA(input_shape, num_classes=config['num_classes'] , dropout_rate=config['dropout_rate'])


time_stamp = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
config['base_path'] = base_path 
config['start_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
logz= f"{config['base_path']}//AttUNet//{config['start_time']}_logs/"


# Callbacks
callbacks = [
    ModelCheckpoint(f"{config['base_path']}//AttUNet//AttUNet_Checkpoint{time_stamp}.h5", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.25, patience=5, min_lr=0.00001),
    EarlyStopping(monitor="val_loss", patience=25, verbose=1),
    WandbCallback()    
]

# Poly Rate scheduler
starter_learning_rate = 0.001
end_learning_rate = 0.00001
decay_steps = 1000
learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    starter_learning_rate,
    decay_steps,
    end_learning_rate,
    power=0.25)



''' 
Copied:
    compile(optimizer=Adam(lr= 1e-2), loss= BinaryFocalLoss(gamma=2), metrics =['accuracy',jacard_coef]   )
'''


# Trying different optimizers
opt1 = tf.keras.optimizers.RMSprop(learning_rate=config['learning_rate'],rho=0.9,momentum=0.9, epsilon=1e-07,centered=True,name="RMSprop")
opt2 = tf.keras.optimizers.Adam(learning_rate=config['learning_rate']) #,amsgrad=True
opt3 = tf.keras.optimizers.SGD(learning_rate=config['learning_rate'], momentum=0.9, nesterov=True, name="SGD")
opt4 = tfa.optimizers.AdamW(weight_decay = 0.001, learning_rate=config['learning_rate'],)

poly_rate = tf.keras.optimizers.SGD(learning_rate = learning_rate_fn)
poly_rate2 = tf.keras.optimizers.Adam(learning_rate = learning_rate_fn)
top_K = 3

start_time = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
print(f'Training start time:{start_time}')

from segmentation_models.losses import bce_jaccard_loss
training_loss_fn = bce_jaccard_loss

# model.compile( optimizer = opt, loss= 'categorical_crossentropy', metrics=['accuracy',tf.keras.metrics.TopKCategoricalAccuracy(top_K, name="top-3-accuracy")])  #label_smoothing=0.05, tf.keras.losses.KLDivergence()
#model.compile( optimizer=opt1,loss="sparse_categorical_crossentropy" , metrics=["sparse_categorical_accuracy"],) #"sparse_categorical_crossentropy" , sparse_categorical_accuracy", tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.3)

if config['num_classes']  > 1:
    model.compile( optimizer=opt4,loss= tf.keras.losses.CategoricalCrossentropy() , metrics=['accuracy',sm.metrics.iou_score] ) # sm.metrics.iou_score,'categorical_crossentropy', sparse_categorical_accuracy", tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.3)
else:
    model.compile( optimizer=opt4,loss= 'binary_crossentropy' , metrics=['accuracy']) #binary_crossentropy, 'binary_crossentropy',sm.metrics.iou_score,

history = model.fit(train_ds, epochs= config['epochs'], validation_data= val_ds, callbacks = callbacks) # , callbacks = callbacks)   mcp_save, callbacks=[reduce_lr_loss]




# model_save_path = f"{ config['base_path'] }/AttentionUNet_Sren/Best_model_{time_stamp}.h5"
# model.save(model_save_path)

plt.plot(history.history['loss'], label = 'Training') ; 
plt.plot(history.history['val_loss'], label = 'Validation'); 
plt.title('Train and Validation loss')
plt.show()

plt.plot(history.history['accuracy'], label = 'Training') ; 
plt.plot(history.history['val_accuracy'], label = 'Validation'); 
plt.title('Train and Validation Accuracy')
plt.show()

# plt.plot(history.history['iou_score'], label = 'Training') ; 
# plt.plot(history.history['val_iou_score'], label = 'Validation'); 
# plt.title('Train and Validation IOU_score')
# plt.legend()
# plt.show()


# Custom colormap
custom_cm = cm.Blues(np.linspace(0,1,30))
custom_cm = colors.ListedColormap(custom_cm[10:,:-1])


model_val_data_path = os.path.join(base_path,'new_test\image\*.mat')
model_val_data = glob.glob(model_val_data_path)

batch_idx = random.randint(1,len(model_val_data)) # Pick any of the default batch

for idx in range(10):
  predict_data = loadmat(model_val_data[batch_idx+idx])
  a0_old,a_gt0 = predict_data['echo_tmp'], predict_data['semantic_seg']
  a0 = np.stack((a0_old,)*3,axis=-1)  
  # a0 = a[idx]
  # a_gt0 = a_gt[idx]
  # ( a0.shape, a_gt0.shape )

  res0 = model.predict ( np.expand_dims(a0,axis=0) ) 
  res0 = res0.squeeze()
  res0_final = np.argmax(res0,axis=2)


  f, axarr = plt.subplots(1,3,figsize=(20,20))

  axarr[0].imshow(a0_old.squeeze(),cmap='gray_r')
  axarr[0].set_title( f'Echo {os.path.basename(model_val_data[batch_idx+idx])}') #.set_text

  axarr[1].imshow(a_gt0.squeeze(),cmap='viridis') # gt
  axarr[1].set_title( 'Ground truth') #.set_text

  axarr[2].imshow(res0_final, cmap='viridis')
  axarr[2].set_title('Prediction')
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
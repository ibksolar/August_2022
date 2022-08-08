# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 18:53:33 2021

@author: i368o351
"""

# coding: utf-8
'''
TODO:
    using attention gate on both spatial and channel dimensions
Dependencies:
    Keras 2.0.8
    Tensorflow 1.3.0
    UNetConfig
'''
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow_addons as tfa
# from tensorflow.contrib.keras import models, layers, regularizers
from tensorflow.keras import backend as K
import random
import os
import numpy as np

from scipy.io import loadmat

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau #TensorBoard
from datetime import datetime

'''
Hyper-parameters
'''

# Data path
data_folder = '/all_block_data/Dec_Train_block_len_21_011121_2331' #all_block_data/Dec_block_len_21_Train_set_291021_1519' # '/all_block_data/Dec_block_len_45_Train_set_181021_1828/'
base_dir = os.path.join ('Y:\ibikunle\Python_Project\Fall_2021' + data_folder ) 

# Confirm path is right...
print(f'{os.path.isdir(base_dir)}')

input_dir = "/image/"
target_dir = "/segment_dir/"

input_img_paths = sorted( os.listdir (base_dir+ input_dir) ) 
target_img_paths = sorted( os.listdir(base_dir + target_dir) ) 

# network structure
filter_num = 64 # number of basic filters for the first layer
filter_size = 3 # shape/size of the convolutional filter

up_samp_size = 2 # size of upsampling filters
SE_RATIO = 2   # reduction ratio of SE block

# input data
# Hyperparameters and constants

image_size_y = 416 #224, 416
image_size_x = 64
img_channels = 1  # 1-grayscale, 3-RGB scale

img_size = (image_size_y,image_size_x) # :( Almost redundant
input_shape = (image_size_y, image_size_x, img_channels)

num_classes = 30
batch_size = 20



# ===========================================================================================
# Echo_Load_Train_Test function
class Echo_Load_Train_Test(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths,base_dir = base_dir ,input_dir = input_dir,target_dir = target_dir, num_classes = num_classes):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.base_dir = base_dir
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.num_classes = num_classes

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        
        # x = np.zeros((self.batch_size,) + self.img_size , dtype="uint8")
        x = np.zeros((self.batch_size,) + self.img_size , dtype="float32") #+ (1,)
        for j, path in enumerate(batch_input_img_paths):
            img_path = base_dir + input_dir + path
            img = loadmat(img_path)
            img = img['echo_tmp']
            img[np.isnan(img)] = 0        
                        
            if np.all(img<=1):
                x[j] = img # np.expand_dims( img, 2) # Normalize /255
            else:
                x[j] = img/255   #np.expand_dims( img/255, 2)
        
        x = np.stack((x,)*3, axis=-1) # SM module requires 3 channels
           
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32") #+ (1,)
        for j, path in enumerate(batch_target_img_paths):
            target_path = base_dir + target_dir + path
            target = loadmat(target_path)
            target = target['semantic_seg'] # raster, semantic_seg
            target[np.isnan(target)] = 0
            # target = target[:IMG_HEIGHT,:IMG_WIDTH]
            target  = ( np.array(target,dtype=bool ) ).astype('float32') #,dtype=bool                        
            y[j] = np.expand_dims( target, 2 )
            
        y = tf.keras.utils.to_categorical(y, num_classes)
        return x, y  


# ===========================================================================================
#               Function
# ===========================================================================================
    
def Create_ListDataLoader( img_paths, base_dir = base_dir, input_dir = input_dir, target_dir = target_dir):
    x_train, y_train = [],[]
    
    for iter,path in enumerate( img_paths ):
                 
      x_new = loadmat(base_dir + input_dir + path)
      x_new = x_new['echo_tmp']
      x_new[np.isnan(x_new)] = 0  
      # x_new = np.expand_dims(x_new,2)
      x_train.append(x_new)   
      
      y_new = loadmat(base_dir + target_dir + path.replace('_dec','_dec_segment'))
      y_new = y_new['semantic_seg']
      y_new[np.isnan(y_new)] = 0
      y_new = y_new[:image_size_y,:image_size_x]  
      # y_new = np.expand_dims(y_new,2)
      y_train.append(y_new)
    
    x_train = np.array(x_train)
    x_train = np.expand_dims(x_train,3)
    
    y_train = np.array(y_train)    
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    print(f'y_train shape after categorical{y_train.shape}')
    
    return (x_train,y_train)


# ===========================================================================================
#              Class Echo_Load_Train_Test2
# ===========================================================================================
# Echo_Load_Train_Test2 function
class Echo_Load_Train_Test2(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths,base_dir = base_dir ,input_dir = input_dir,target_dir = target_dir, num_classes = num_classes):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.base_dir = base_dir
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.num_classes = num_classes

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        
        # x = np.zeros((self.batch_size,) + self.img_size , dtype="uint8")
        x = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img_path = base_dir + input_dir + path
            img = loadmat(img_path)
            img = img['echo_tmp']
            img[np.isnan(img)] = 0
            
            if np.all(img<=1):
                x[j] = np.expand_dims( img, 2) # Normalize /255
            else:
                x[j] = np.expand_dims( img/255, 2)

        # y = np.zeros((self.batch_size,) + self.img_size , dtype="uint8")    
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_target_img_paths):
            target_path = base_dir + target_dir + path
            target = loadmat(target_path)
            target = target['semantic_seg'] # raster
            target[np.isnan(target)] = 0
            target = target[:image_size_y,:image_size_x]
            target = ( np.array(target) ).astype('int') #,dtype=bool                        
            y[j] = np.expand_dims( target, 2 )
        y = tf.keras.utils.to_categorical(y, num_classes)
        return x, y  

train_samples = round(0.768* len(input_img_paths)  ) #1000 # 1000        
val_samples = 250 # 500
test_samples = len(input_img_paths) - train_samples - val_samples

random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:train_samples] # input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:train_samples] # target_img_paths[:-val_samples]

val_input_img_paths = input_img_paths[train_samples:train_samples+val_samples+1] # input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[train_samples:train_samples+val_samples+1]

# Instantiate training and testing data
train_gen = Echo_Load_Train_Test2(batch_size, img_size, train_input_img_paths, train_target_img_paths)
val_gen = Echo_Load_Train_Test2(batch_size, img_size, val_input_img_paths, val_target_img_paths)

if test_samples > 1:
    test_input_img_paths = input_img_paths[-test_samples:] # input_img_paths[-val_samples:]
    test_target_img_paths = target_img_paths[-test_samples:]
    test_gen = Echo_Load_Train_Test2(batch_size, img_size, test_input_img_paths, test_target_img_paths)
    

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
    axis = 3
    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=axis)(conv)
    conv = layers.Activation('relu')(conv)
    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(conv) # Sreeni removed this
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=axis)(conv)
    conv = layers.Activation('relu')(conv)
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)
    
    # Sreeni did not add this part ( he returned conv from here (Error - this was another fxn))
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
    :param inter_shape: intermedium channle numer
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


def Attention_ResUNet_PA(dropout_rate=0.0, batch_norm=True):
    '''
    Rsidual UNet construction, with attention gate
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
    inputs = layers.Input(shape = input_shape, dtype=tf.float32)
    axis = 3

    # Downsampling layers
    # DownRes 1, double residual convolution + pooling
    conv_128 = double_conv_layer(inputs, filter_size, filter_num, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_64 = double_conv_layer(pool_64, filter_size, 2*filter_num, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = double_conv_layer(pool_32, filter_size, 4*filter_num, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = double_conv_layer(pool_16, filter_size, 8*filter_num, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = double_conv_layer(pool_8, filter_size, 16*filter_num, dropout_rate, batch_norm)

    # Upsampling layers

    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    # channel attention block
    se_conv_16 = SE_block(conv_16, out_dim=8*filter_num, ratio=SE_RATIO, name='att_16')
    # spatial attention block
    gating_16 = gating_signal(conv_8, 8*filter_num, batch_norm)
    att_16 = attention_block(se_conv_16, gating_16, 8*filter_num, name='att_16')
    # attention re-weight & concatenate
    up_16 = layers.UpSampling2D(size=(up_samp_size, up_samp_size), data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=axis)
    up_conv_16 = double_conv_layer(up_16, filter_size, 8*filter_num, dropout_rate, batch_norm)

    # UpRes 7
    # channel attention block
    se_conv_32 = SE_block(conv_32, out_dim=4*filter_num, ratio=SE_RATIO, name='att_32')
    # spatial attention block
    gating_32 = gating_signal(up_conv_16, 4*filter_num, batch_norm)
    att_32 = attention_block(se_conv_32, gating_32, 4*filter_num, name='att_32')
    # attention re-weight & concatenate
    up_32 = layers.UpSampling2D(size=(up_samp_size, up_samp_size), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=axis)
    up_conv_32 = double_conv_layer(up_32, filter_size, 4*filter_num, dropout_rate, batch_norm)

    # UpRes 8
    # channel attention block
    se_conv_64 = SE_block(conv_64, out_dim=2*filter_num, ratio=SE_RATIO, name='att_64')
    # spatial attention block
    gating_64 = gating_signal(up_conv_32, 2*filter_num, batch_norm)
    att_64 = attention_block(se_conv_64, gating_64, 2*filter_num, name='att_64')
    # attention re-weight & concatenate
    up_64 = layers.UpSampling2D(size=(up_samp_size, up_samp_size), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, att_64], axis=axis)
    up_conv_64 = double_conv_layer(up_64, filter_size, 2*filter_num, dropout_rate, batch_norm)

    # UpRes 9
    # channel attention block
    se_conv_128 = SE_block(conv_128, out_dim=filter_num, ratio=SE_RATIO, name='att_128')
    # spatial attention block
    gating_128 = gating_signal(up_conv_64, filter_num, batch_norm)
    # attention re-weight & concatenate
    att_128 = attention_block(se_conv_128, gating_128, filter_num, name='att_128')
    up_128 = layers.UpSampling2D(size=(up_samp_size, up_samp_size), data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, att_128], axis=axis)
    up_conv_128 = double_conv_layer(up_128, filter_size, filter_num, dropout_rate, batch_norm)
    
    # up_conv_128 = tf.reduce_sum(up_conv_128,axis=-1)
    up_conv_128 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up_conv_128)
 
    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(up_conv_128) #sigmoid, softmax
    
    # 1*1 convolutional layers
    # valid padding
    # batch normalization
    # sigmoid nonlinear activation
    conv_final = layers.Conv2D(num_classes, kernel_size=(1,1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=axis)(conv_final)
    conv_final = layers.Activation('softmax')(conv_final) #relu

    # Model integration
    model = keras.Model(inputs, outputs, name="AttentionSEResUNet")
    return model

# Instantiate a model
model = Attention_ResUNet_PA(dropout_rate=0.1)

learning_rate = 3e-3
num_epochs = 30



# Callbacks
callbacks = [
    ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.25, patience=5, min_lr=0.0001),
    #EarlyStopping(monitor="val_loss", patience=25, verbose=1), WandbCallback()
    
]

# Poly Rate scheduler
starter_learning_rate = 0.001
end_learning_rate = 0.0001
decay_steps = 1000
learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    starter_learning_rate,
    decay_steps,
    end_learning_rate,
    power=0.25)


# Trying different optimizers
opt1 = tf.keras.optimizers.RMSprop(learning_rate=learning_rate,rho=0.9,momentum=0.9, epsilon=1e-07,centered=True,name="RMSprop")
opt2 = tf.keras.optimizers.Adam(learning_rate=learning_rate,amsgrad=True)
opt3 = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True, name="SGD")
opt4 = tfa.optimizers.AdamW(weight_decay = 0.001, learning_rate=learning_rate,)

poly_rate = tf.keras.optimizers.SGD(learning_rate = learning_rate_fn)
poly_rate2 = tf.keras.optimizers.Adam(learning_rate = learning_rate_fn)
top_K = 3

start_time = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
print(f'Training start time:{start_time}')

# model.compile( optimizer = opt, loss= 'categorical_crossentropy', metrics=['accuracy',tf.keras.metrics.TopKCategoricalAccuracy(top_K, name="top-3-accuracy")])  #label_smoothing=0.05, tf.keras.losses.KLDivergence()
#model.compile( optimizer=opt1,loss="sparse_categorical_crossentropy" , metrics=["sparse_categorical_accuracy"],) #"sparse_categorical_crossentropy" , sparse_categorical_accuracy", tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.3)
model.compile( optimizer=opt4,loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2) , metrics=['accuracy',tf.keras.metrics.TopKCategoricalAccuracy(top_K, name="top-3-accuracy")],) # sparse_categorical_accuracy", tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.3)


history = model.fit(train_gen,  epochs= num_epochs, validation_data= val_gen) # , callbacks = callbacks)   mcp_save, callbacks=[reduce_lr_loss]




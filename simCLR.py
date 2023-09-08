# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 23:17:36 2022
First Unsupervised try

@author: i368o351
"""



from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from tqdm import tqdm
import cv2

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

import os

from custom_loadmat import loadmat
import glob
from scipy.ndimage import median_filter as sc_med_filt
from datetime import datetime

# from keras.metrics import MeanIoU
# from sklearn.metrics import roc_auc_score
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,ReduceLROnPlateau
# from focal_loss import SparseCategoricalFocalLoss
# import random
# import tensorflow_addons as tfa
# from matplotlib import cm,colors


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


use_wandb = True
time_stamp = '26_November_22_0218' # datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')

if use_wandb:    
    ## WandB config
    import wandb
    # from wandb.keras import WandbCallback  
    wandb.init( project="my-test-project", entity="ibksolar", name='SimCLR_dec_data'+time_stamp,config ={})
    config = wandb.config
else:
    config ={}



# PATHS
# Path to data
base_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data'  # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'
#train_aug = os.path.join(base_path,'augmented_plus_train_data\*.mat')
train_path = os.path.join(base_path,'train_data\*.mat')
val_path = os.path.join(base_path,'val_data\*.mat')
test_path = os.path.join(base_path,'test_data\*.mat')   

# Create tf.data.Dataset
config['Run_Note'] = 'First try of Unsupervised'
config['batch_size'] = 4

# config['img_y'] = 416 #1664 #1664 , 416
# config['img_x'] = 256 #256, 64
config['img_y'],config['img_x'] = loadmat(glob.glob(train_path)[0])['echo_tmp'].shape



# Training params
#config={}
config['img_channels'] = 3

config['num_classes'] = 1
config['epochs'] = 200
config['learning_rate'] = 1e-4
config['base_path'] = base_path
SEED = 42
AUTO = tf.data.experimental.AUTOTUNE



# =============================================================================
# Function for training data
def read_mat_single(filepath):
    def _read_mat(filepath):
        
        filepath = bytes.decode(filepath.numpy())      
        mat_file = loadmat(filepath)
        echo = tf.cast(mat_file['echo_tmp'], dtype=tf.float64)
        
        echo = tf.expand_dims(echo, axis=-1)
        
        if config['img_channels']> 1:            
            echo = tf.image.grayscale_to_rgb(echo)         
        
        shape0 = echo.shape        
        return echo,np.asarray(shape0)     
    
    output = tf.py_function(_read_mat,[filepath],[tf.double, tf.int64])
    shape = output[1]
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([config['img_y'],config['img_x'], config['img_channels'] ])   
    return data0

# =============================================================================
##  
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


train_ds = tf.data.Dataset.list_files(train_path,shuffle=True) #'*.mat'
train_ds = train_ds.map(read_mat_single,num_parallel_calls=8)
train_ds = train_ds.batch(config['batch_size'],drop_remainder=True).prefetch(AUTO) #.shuffle(buffer_size = 100 * config['batch_size'])

# No augmentation for testing and validation
val_ds = tf.data.Dataset.list_files(val_path,shuffle=True)
val_ds = val_ds.map(read_mat_single,num_parallel_calls=8)
val_ds = val_ds.batch(config['batch_size'],drop_remainder=True).cache().prefetch(AUTO)

test_ds = tf.data.Dataset.list_files(test_path,shuffle=True)
test_ds = test_ds.map(read_mat_single,num_parallel_calls=8)
test_ds = test_ds.batch(config['batch_size'],drop_remainder=True).cache().prefetch(AUTO)


#=============================================================================
# Custom Augment ( May add this to dataloader later)
#=============================================================================
class CustomAugment(object):
    def __call__(self, sample):        
        # Random flips
        sample = self._random_apply(tf.image.flip_left_right, sample, p=0.5)
        
        # Randomly apply transformation (color distortions) with probability p.
        sample = self._random_apply(self._color_jitter, sample, p=0.8)
        sample = self._random_apply(self._color_drop, sample, p=0.2)

        return sample

    def _color_jitter(self, x, s=1):
        # one can also shuffle the order of following augmentations
        # each time they are applied.
        x = tf.image.random_brightness(x, max_delta=0.8*s)
        x = tf.image.random_contrast(x, lower=1-0.8*s, upper=1+0.8*s)
        x = tf.image.random_saturation(x, lower=1-0.8*s, upper=1+0.8*s)
        x = tf.image.random_hue(x, max_delta=0.2*s)
        x = tf.clip_by_value(x, 0, 1)
        return x
    
    def _color_drop(self, x):
        x = tf.image.rgb_to_grayscale(x)
        x = tf.tile(x, [1, 1, 1, 3])
        return x
    
    def _random_apply(self, func, x, p):
        return tf.cond(
          tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                  tf.cast(p, tf.float32)),
          lambda: func(x),
          lambda: x)

data_augmentation = Sequential([Lambda(CustomAugment())])


input_shape = (config['img_y'], config['img_x'], config['img_channels'])
dtype_used = tf.float64

#=============================================================================
# Utilities
#=============================================================================
def get_resnet_simclr(hidden_1, hidden_2, hidden_3):
    base_model = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape= input_shape )
    base_model.trainable = True
    inputs = Input(input_shape, dtype = dtype_used)
    h = base_model(inputs, training=True)
    h = tf.cast( GlobalAveragePooling2D()(h) , dtype = dtype_used)

    projection_1 = Dense(hidden_1, dtype = dtype_used)(h)
    projection_1 = Activation("relu")(projection_1)
    projection_2 = Dense(hidden_2, dtype = dtype_used)(projection_1)
    projection_2 = Activation("relu")(projection_2)
    projection_3 = Dense(hidden_3, dtype = dtype_used)(projection_2)

    resnet_simclr = Model(inputs, projection_3)

    return resnet_simclr

#=============================================================================
# Helper
#=============================================================================
# from augmentation.gaussian_filter import GaussianBlur

def get_negative_mask(batch_size):
    # return a mask that removes the similarity score of equal/similar images.
    # this function ensures that only distinct pair of images get their similarity scores
    # passed as negative examples
    negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0
    return tf.constant(negative_mask)


def gaussian_filter(v1, v2):
    k_size = int(v1.shape[1] * 0.1)  # kernel size is set to be 10% of the image height/width
    gaussian_ope = GaussianBlur(kernel_size=k_size, min=0.1, max=2.0)
    [v1, ] = tf.py_function(gaussian_ope, [v1], [tf.float32])
    [v2, ] = tf.py_function(gaussian_ope, [v2], [tf.float32])
    return v1, v2

#=============================================================================
# Losses
#=============================================================================
cosine_sim_1d = tf.keras.losses.CosineSimilarity(axis=1, reduction=tf.keras.losses.Reduction.NONE)
cosine_sim_2d = tf.keras.losses.CosineSimilarity(axis=2, reduction=tf.keras.losses.Reduction.NONE)


def _cosine_simililarity_dim1(x, y):
    v = cosine_sim_1d(x, y)
    return v

def _cosine_simililarity_dim2(x, y):
    # x shape: (N, 1, C)
    # y shape: (1, 2N, C)
    # v shape: (N, 2N)
    v = cosine_sim_2d(tf.expand_dims(x, 1), tf.expand_dims(y, 0))
    return v

def _dot_simililarity_dim1(x, y):
    # x shape: (N, 1, C)
    # y shape: (N, C, 1)
    # v shape: (N, 1, 1)
    v = tf.matmul(tf.expand_dims(x, 1), tf.expand_dims(y, 2))
    return v


def _dot_simililarity_dim2(x, y):
    v = tf.tensordot(tf.expand_dims(x, 1), tf.expand_dims(tf.transpose(y), 0), axes=2)
    # x shape: (N, 1, C)
    # y shape: (1, C, 2N)
    # v shape: (N, 2N)
    return v
#=============================================================================

class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

negative_mask = get_negative_mask(config['batch_size'])


#=============================================================================
#=============================================================================
@tf.function
def train_step(xis, xjs, model, optimizer, criterion, temperature):
    with tf.GradientTape() as tape:
        zis = model(xis)
        zjs = model(xjs)

        # normalize projection feature vectors
        zis = tf.math.l2_normalize(zis, axis=1)
        zjs = tf.math.l2_normalize(zjs, axis=1)

        l_pos = _dot_simililarity_dim1(zis, zjs)
        l_pos = tf.reshape(l_pos, (config['batch_size'], 1))
        l_pos /= temperature

        negatives = tf.concat([zjs, zis], axis=0)

        loss = 0

        for positives in [zis, zjs]:
            l_neg = _dot_simililarity_dim2(positives, negatives)

            labels = tf.zeros(config['batch_size'], dtype=tf.int32)

            l_neg = tf.boolean_mask(l_neg, negative_mask)
            l_neg = tf.reshape(l_neg, (config['batch_size'], -1))
            l_neg /= temperature

            logits = tf.concat([l_pos, l_neg], axis=1) 
            loss += criterion(y_pred=logits, y_true=labels)

        loss = loss / (2 * config['batch_size'])

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss
#=============================================================================
def train_simclr(model, dataset, optimizer, criterion,
                 temperature=0.2, epochs=100):
    step_wise_loss = []
    epoch_wise_loss = []

    for epoch in tqdm(range(epochs)):
        for image_batch in dataset:
            a = data_augmentation(image_batch)
            b = data_augmentation(image_batch)

            loss = train_step(a, b, model, optimizer, criterion, temperature)
            step_wise_loss.append(loss)

        epoch_wise_loss.append(np.mean(step_wise_loss))
        wandb.log({"nt_xentloss": np.mean(step_wise_loss)})
        
        if epoch % 10 == 0:
            print("epoch: {} loss: {:.3f}".format(epoch + 1, np.mean(step_wise_loss)))

    return epoch_wise_loss, model
#=============================================================================


#=============================================================================
# Training
#=============================================================================
criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, 
                                                          reduction=tf.keras.losses.Reduction.SUM)
decay_steps = 1000
lr_decayed_fn = tf.keras.experimental.CosineDecay(
    initial_learning_rate=0.1, decay_steps=decay_steps)
optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)

resnet_simclr_2 = get_resnet_simclr(256, 128, 64)

epoch_wise_loss, resnet_simclr  = train_simclr(resnet_simclr_2, train_ds, optimizer, criterion,
                 temperature=0.1, epochs=400)

with plt.xkcd():
    plt.plot(epoch_wise_loss)
    plt.title("tau = 0.1, h1 = 256, h2 = 128, h3 = 64")
    plt.show()
criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, 
                                                          reduction=tf.keras.losses.Reduction.SUM)


# Supervised_fine_tuninig
logz= f"{config['base_path']}/SimCLR/{config['start_time']}_logs/"
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,ReduceLROnPlateau
callbacks = [ ModelCheckpoint(f"{config['base_path']}//SimCLR//SimCLR_Checkpoint{time_stamp}.h5", save_best_only=True, monitor="val_loss"), 
             TensorBoard(log_dir = logz,histogram_freq = 1,profile_batch = '1,70', embeddings_freq=50),
             ReduceLROnPlateau(monitor="val_loss", factor=0.25, patience=10, min_lr=0.00005, verbose= 1),
             EarlyStopping(monitor="val_loss", patience=30, verbose=1)]


resnet_simclr.fit(train_ds,val_ds,epoch=150,callbacks = callbacks)









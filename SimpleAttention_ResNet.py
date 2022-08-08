# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 21:37:51 2021

@author: i368o351
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow_addons as tfa
# from tensorflow.contrib.keras import models, layers, regularizers
#from tensorflow.keras import backend as K
import random
import os
import numpy as np

from scipy.io import loadmat

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau #TensorBoard
from datetime import datetime

import wandb
from wandb.keras import WandbCallback
wandb.init(project="Nov2021_Segmentation_Attentions", entity="ibksolar", name="SimpleAttention_ResNet")


#tf.keras.backend.set_floatx('float16')
#tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy('mixed_float16')
'''
Hyper-parameters
'''

# Data path
data_folder = '/all_block_data/Dec_Train_block_len_21_011121_2331' #all_block_data/Dec_block_len_21_Train_set_291021_1519' # '/all_block_data/Dec_block_len_45_Train_set_181021_1828/'
base_dir = os.path.join ('Y:\ibikunle\Python_Project\Fall_2021' + data_folder ) 

# Confirm path is right...
print(f'{os.path.isdir(base_dir)}')

input_dir = "/image/"
target_dir = "/raster_dir/" #segment_dir

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
input_shape = (image_size_y, image_size_x,img_channels ) #img_channels

num_classes = 1
batch_size = 20


# ===========================================================================================
# Echo_Load_Train_Test function
class Echo_Load_Train_Test(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths,base_dir = base_dir ,input_dir = input_dir,
                 target_dir = target_dir, num_classes = num_classes):
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
           
        # Initialize y depending on if it's Binary or Multi-class
        
        
        
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32") #+ (1,)
        # if num_classes <= 1:    
        # else:
        #     y = np.zeros((self.batch_size,) + self.img_size, dtype="float16") #+ (1,)
        
        for j, path in enumerate(batch_target_img_paths):
            target_path = base_dir + target_dir + path
            target = loadmat(target_path)
            target = target['raster'] # raster, semantic_seg
            target[np.isnan(target)] = 0
            # target = target[:IMG_HEIGHT,:IMG_WIDTH]
            
            if num_classes <= 1:
                target  = ( np.array(target,dtype=bool ) ).astype('float16') #,dtype=bool
            else:
                target  = ( np.array(target) ).astype('float16') #,dtype=bool
        
            #if num_classes <= 1:
            y[j] = np.expand_dims( target, 2 )            
            
        if num_classes > 1:  
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
train_gen = Echo_Load_Train_Test(batch_size, img_size, train_input_img_paths, train_target_img_paths)
val_gen = Echo_Load_Train_Test(batch_size, img_size, val_input_img_paths, val_target_img_paths)

if test_samples > 1:
    test_input_img_paths = input_img_paths[-test_samples:] # input_img_paths[-val_samples:]
    test_target_img_paths = target_img_paths[-test_samples:]
    test_gen = Echo_Load_Train_Test(batch_size, img_size, test_input_img_paths, test_target_img_paths)


# Network Architecture and functions

def ResNetBlock(x,nodes):
    
    #x =   layers.Conv2D(filters=64, kernel_size=3, padding="same")(x) #input_layer
    conv1 = layers.Conv2D(filters=64, kernel_size=3, padding="same",kernel_initializer='he_normal')(x) # input_layer, Conv1D
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.ReLU()(conv1)

    conv2 = layers.Conv2D(filters=64, kernel_size=3, padding="same",kernel_initializer='he_normal')(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.ReLU()(conv2)

    conv3 = layers.Conv2D(filters=64, kernel_size=3, padding="same",kernel_initializer='he_normal')(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.ReLU()(conv3)
    
    conv4 = layers.Conv2D(filters=64, kernel_size=3, padding="same",kernel_initializer='he_normal')(conv3)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.ReLU()(conv4)
    
    conv5 = layers.Conv2D(filters=nodes, kernel_size=3, padding="same",kernel_initializer='he_normal')(conv4)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.ReLU()(conv5)
    
    conv5 = layers.add([x,conv5])
    x = layers.ReLU()(conv3) # Overwrite x
    
    return x

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv2D(filters=ff_dim, kernel_size=(5,3),padding="same", activation="relu",kernel_initializer='he_normal')(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv2D(filters=ff_dim, kernel_size=(5,3),padding="same", activation="relu",kernel_initializer='he_normal')(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    nodes,
    dropout=0,
    mlp_dropout=0,    
):
    inputs = tf.keras.Input(shape=input_shape) # shape=input_shape
    x = inputs
    #x =   layers.Conv2D(filters=64, kernel_size=3, padding="same")(inputs)   
     
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)    
    
    for _ in range(resnet_heads):
        x = ResNetBlock(x,nodes = 1)
        
    x = tf.reduce_sum(x,axis=-1)
    # Pool or average
    # x = layers.GlobalAveragePooling1D()(x)#data_format="channels_first"    
    # x = layers.Flatten()(x)   
    
    # for dim in mlp_units:
    #     x = layers.Dense(dim, activation="relu")(x)
    #     x = layers.Dropout(mlp_dropout)(x)   
    
    # x = tf.reduce_sum(x,axis=-1) # Add in "filters" dimension to speed up?
    x = layers.LayerNormalization(epsilon=1e-6)(x)   
   
    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)


 #input_shape = (21,5,) #x_train.shape[2]

# ResNet params
resnet_heads = 5

# Attention params
num_epochs = 30

nodes = 64 #512
learning_rate = 1e-3

head_size = 64 # 256,64,128
num_heads = 4 #15
ff_dim= 1
num_transformer_blocks= 25
mlp_units= [128]  # 128
mlp_dropout=0.3     #0.4
dropout=0.2          #0.25

model = build_model(input_shape,head_size=head_size,num_heads=num_heads,ff_dim=ff_dim,
                    num_transformer_blocks = num_transformer_blocks,
                    mlp_units=mlp_units, nodes=nodes, 
                    mlp_dropout=mlp_dropout, dropout=dropout)

wandb.config = {
  "learning_rate": "learning_rate",
  "epochs": num_epochs, 
  "batch_size": batch_size,
  "nodes": nodes,
  "row_length":image_size_x,
  "base_path":base_dir,
  "head_size":head_size, "num_heads":num_heads,
  "ff_dim":ff_dim, "num_transformer_blocks":num_transformer_blocks,
  "mlp_units":mlp_units, "mlp_dropout":mlp_dropout,
  "dropout":dropout
}


# Poly Rate scheduler
starter_learning_rate = 0.001
end_learning_rate = 0.0001
decay_steps = 1000
learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    starter_learning_rate,
    decay_steps,
    end_learning_rate,
    power=0.25)

# Callbacks
callbacks = [
    ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.25, patience=5, min_lr=0.0001),
    #EarlyStopping(monitor="val_loss", patience=25, verbose=1),  
    WandbCallback()   
]

# Trying different optimizers
opt1 = tf.keras.optimizers.RMSprop(learning_rate=learning_rate,rho=0.9,momentum=0.9, epsilon=1e-07,centered=True,name="RMSprop")
opt2 = tf.keras.optimizers.Adam(learning_rate=learning_rate,amsgrad=True)
opt3 = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True, name="SGD")
opt4 = tfa.optimizers.AdamW(weight_decay = 0.0001, learning_rate=learning_rate,)

poly_rate = tf.keras.optimizers.SGD(learning_rate = learning_rate_fn)
poly_rate2 = tf.keras.optimizers.Adam(learning_rate = learning_rate_fn)
top_K = 3

start_time = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
print(f'Training start time:{start_time}')

# model.compile( optimizer = opt, loss= 'categorical_crossentropy', metrics=['accuracy',tf.keras.metrics.TopKCategoricalAccuracy(top_K, name="top-3-accuracy")])  #label_smoothing=0.05, tf.keras.losses.KLDivergence()
#model.compile( optimizer=opt1,loss="sparse_categorical_crossentropy" , metrics=["sparse_categorical_accuracy"],) #"sparse_categorical_crossentropy" , sparse_categorical_accuracy", tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.3)
model.compile( optimizer=opt4,loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2) , metrics=['accuracy',tf.keras.metrics.TopKCategoricalAccuracy(top_K, name="top-3-accuracy")],) # sparse_categorical_accuracy", tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.3)


history = model.fit(train_gen,
          epochs= num_epochs, 
          #batch_size= batch_size, 
          validation_data=val_gen,
         callbacks=callbacks) #mcp_save, callbacks=[reduce_lr_loss]





def myprint(s):
    with open('modelsummary.txt','w+') as f:
        print(s, file=f)

if 0:
    model.summary(print_fn=myprint)















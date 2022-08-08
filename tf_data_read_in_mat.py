# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 12:49:13 2022

@author: i368o351
"""

from tensorflow.keras import layers
from tensorflow import keras
from keras import backend as K

import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

import os
import random
from scipy.io import loadmat
from scipy.ndimage import median_filter as sc_med_filt

from keras.metrics import MeanIoU

import glob
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,ReduceLROnPlateau
#import segmentation_models as sm
from datetime import datetime



data_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\image\*.mat'

def read_mat(filepath):
    def _read_mat(filepath):
        
        filepath = bytes.decode(filepath.numpy())      
        mat_file = loadmat(filepath)
        echo = mat_file['echo_tmp']
        layer = mat_file['semantic_seg']        
        return echo,layer 
    output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double])
    
    return output[0],output[1]



ds = tf.data.Dataset.list_files(data_path) #'*.mat'
ds = ds.map(read_mat,num_parallel_calls=8)
ds = ds.batch(4,drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

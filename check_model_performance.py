# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 20:33:12 2021

@author: i368o351
"""

from __future__ import print_function

import os
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.models import  Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,ReduceLROnPlateau

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.activations import relu
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

import tensorflow_addons as tfa

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from scipy.io import loadmat 
import mat73
from datetime import datetime
# import ipynbname

print("Packages Loaded")


base_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Old_data\Dec_Train_block_len_21_231121_1531'

# Confirm path is right...
print(f'{os.path.isdir(base_path)}')

## Load data

raw_data1 = mat73.loadmat(base_path + '/echo_cnn_in_out_jstars.mat')
all_data = raw_data1['echo_cnn_input']
all_target = raw_data1['echo_cnn_target']
all_coords = raw_data1['coords']
echo_idx = raw_data1['orig_echo_idx']

# Set all nan in the data to zero
nan_idx = np.isnan(all_data).any(axis =-1)
all_target[nan_idx] = 0
all_data[ np.isnan(all_data) ]= 0


## Truncate data because data after truncate point is notgood for training
echo_idx = np.asarray(echo_idx)
stop_val = 600

stop_list, = np.where(echo_idx == stop_val)
stop_idx = stop_list[-1]

difficult_data = all_data[stop_idx+1:]
y_difficult = raw_data1['echo_cnn_target'][stop_idx+1:]


all_data = all_data[:stop_idx]
all_target = all_target[:stop_idx]

print(f'Data shape {all_data.shape}')
print(f'Target shape {all_target.shape}')



row_length = 21 # CHANGE HERE <==
col_length = 15
neigh = 4
mid_pt = 8

# Check that the dimension of data is correct
if all_data.shape[1] == row_length*col_length:
    print('Dimensions match')
else:
    print(f' Row block length:{row_length} and col length:{col_length} does not match Data dimension:{all_data.shape[1]}')     
  
max_class = row_length 
# Highest class is mapped to row_length+1
all_target[all_target == max_class+1 ] = 0


shuffle = 1
if shuffle:
    random.Random(1337).shuffle(all_data)
    random.Random(1337).shuffle(all_target)
    # random.Random(1337).shuffle(all_coord_exp)
    # random.Random(1337).shuffle(all_idx)

## Prep data
train_size = int(np.floor(0.8*len(all_target)));
test_size = int(np.round( 0.15* all_data.shape[0] ))
val_size = all_data.shape[0] -train_size - test_size


x_train = all_data[0:train_size,:]
x_train = np.reshape( x_train, (x_train.shape[0],max_class,-1) )
x_train = x_train[:,:,mid_pt-neigh:mid_pt+neigh+1]
# x_train = np.reshape(x_train,(x_train.shape[0],-1))
x_train = np.transpose(x_train,(0,2,1))
# coords_train = all_coord_exp[0:train_size,:]


x_test = all_data[train_size:train_size+test_size,:]
x_test = np.reshape( x_test,(x_test.shape[0],max_class,-1) )
x_test = x_test[:,:,mid_pt-neigh:mid_pt+neigh+1]
# x_test = np.reshape(x_test,(x_test.shape[0],-1))
x_test = np.transpose(x_test,(0,2,1))
# coords_test = all_coord_exp[train_size:train_size+test_size,:]


x_val_default = all_data[-val_size:,:]
x_val = np.reshape( x_val_default,(x_val_default.shape[0],max_class,-1) )
x_val = x_val[:,:,mid_pt-neigh:mid_pt+neigh+1]
# x_val = np.reshape(x_val,(x_val.shape[0],-1))
x_val = np.transpose(x_val,(0,2,1))
# coords_val = all_coord_exp[-val_size:,:]


y_train = all_target[:train_size]
y_test  = all_target[train_size:train_size+test_size]
y_val = all_target[-val_size:]

var_input_shape = x_train.shape[1:] # 240 columns
num_classes = max_class+1 # layers


# Convert labels to categorical orthonormal vectors
y_train_1hot = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_1hot  = tf.keras.utils.to_categorical(y_test, num_classes)

print(f'Shape of X_train:{x_train.shape}  X_test:{x_test.shape}')
print(f'Shape of y_train:{y_train.shape}  y_test:{y_test.shape}')



## Import Models 
Conv1_model_23_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Old_data\Dec_Train_block_len_21_231121_1531\NewConv1_model\best_model.h5'
NewResNet_23_path  = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Old_data\Dec_Train_block_len_21_231121_1531\NewResNet_weights_3ConvBlocks\29_November_21_18_50_best_model.h5'

Newconv1_13_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Old_data\Dec_Train_block_len_21_131121_2213\NewConv1_model\best_model.h5'
NewAttention_13_ResNet1_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Old_data\Dec_Train_block_len_21_131121_2213\NewResNet_weights_3ConvBlocks\NewResNet_Nov21\NewConv1_model_19_November_21_12_04_Acc_0.6530701518058777_21x15.h5'
CCTRowBlock_13_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Old_data\Dec_Train_block_len_21_131121_2213\CCT_RowBlock\RowBlockAcc_ 13.38_Top5 52.83_Epoch50_18_November_21_1100.h5'

VOldResNetCNN_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Dec_block_len_21_Train_set_291021_151901_November_21_0853_weight_21x15.h5'

NewRowBlockLSTM_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Old_data\Dec_Train_block_len_21_131121_2213\NewAttention_RowBlockLSTM1\30_November_21_1044_Acc_0.685_Top3Acc0.885_21x9.h5'


# Trained models
Newconv1_13_model = tf.keras.models.load_model(Newconv1_13_path) # shape = 21x5
Newconv1_13_model.output_names = 'NewResNet_23_model'


NewAttention_13_ResNet1_model = tf.keras.models.load_model(NewAttention_13_ResNet1_path) # shape = 21x5
NewAttention_13_ResNet1_model.output_names = 'NewAttention_13_ResNet1_model'

NewRowBlockLSTM_path_model = tf.keras.models.load_model(NewRowBlockLSTM_path) # shape = 9x21x1
NewRowBlockLSTM_path_model.output_names = 'NewRowBlockLSTM_path_model'

#CCTRowBlock_13_model = tf.keras.models.load_model(CCTRowBlock_13_path,custom_objects={'CCTTokenizer':CCTTokenizer,'StochasticDepth':StochasticDepth}) 

VOldResNetCNN_model = tf.keras.models.load_model(VOldResNetCNN_path) ## shape = 315x1
VOldResNetCNN_model.output_names = 'ResNetCNNweights_13_model'

Conv1_model_23_model = tf.keras.models.load_model(Conv1_model_23_path)
Conv1_model_23_model.output_names = 'Conv1_model_23_model'

NewResNet_23_model = tf.keras.models.load_model(NewResNet_23_path)
NewResNet_23_model.output_names = 'NewResNet_23_model'


## Test models on val data

conv_xval = np.reshape(x_val_default,(x_val_default.shape[0],row_length,-1))
conv_xval = conv_xval[:,:,6:11]

conv_xval_diff = np.reshape(difficult_data,(difficult_data.shape[0],row_length,-1))
conv_xval_diff = conv_xval_diff[:,:,6:11]
y_difficult[ y_difficult == max_class+1] = 0


def Test_conv_models(curr_model,start_idx=0,diff=False):
    if start_idx == 0:
        start_idx = random.randint(0,len(x_val))
    print(f' Random start index for model: {curr_model.output_names} is {start_idx} ') 
    
    if diff:
        y_pred_val = [ (int(y_difficult[idx]), np.argmax(curr_model.predict(np.expand_dims(conv_xval_diff[idx],axis=0))) ) for idx in range(start_idx,start_idx+200) ]
    else:      
        y_pred_val = [ (int(y_val[idx]), np.argmax(curr_model.predict(np.expand_dims(conv_xval[idx],axis=0))) ) for idx in range(start_idx,start_idx+200) ] #len(x_val)
    
    val_exact_accuracy = 100 * sum([1 if y_pred_val[idx][0]==y_pred_val[idx][1]  else 0 for idx in range(len(y_pred_val)) ]) / len(y_pred_val)
    val_margin5_accuracy = 100 * sum([1 if abs(y_pred_val[idx][0]- y_pred_val[idx][1] ) < 5 else 0 for idx in range(len(y_pred_val)) ]) / len(y_pred_val)    
    print(f' Exact accuracy is {val_exact_accuracy: .5f} %')
    print(f' Top 5% accuracy is {val_margin5_accuracy: .5f} %')
    
    return y_pred_val


def other_models(curr_model,start_idx,diff=False):
    if start_idx == 0:
        start_idx = random.randint(0,len(x_val))
    print(f' Random start index for model: {curr_model.output_names} is {start_idx} ') 
    
    if diff:
        # TO DO: The data shape isn't right 
        y_pred_val = [ (int(y_difficult[idx]), np.argmax(curr_model.predict(np.expand_dims(conv_xval_diff[idx],axis=0))) ) for idx in range(start_idx,start_idx+200) ] 
    else:
        y_pred_val = [ (int(y_val[idx]), np.argmax(curr_model.predict(np.expand_dims(x_val[idx],axis=0))) ) for idx in range(start_idx,start_idx+200) ] #len(x_val)
    
    val_exact_accuracy = 100 * sum([1 if y_pred_val[idx][0]==y_pred_val[idx][1]  else 0 for idx in range(len(y_pred_val)) ]) / len(y_pred_val)
    val_margin5_accuracy = 100 * sum([1 if abs(y_pred_val[idx][0]- y_pred_val[idx][1] ) < 5 else 0 for idx in range(len(y_pred_val)) ]) / len(y_pred_val)    
    print(f' Exact accuracy is {val_exact_accuracy: .5f} %')
    print(f' Top 5% accuracy is {val_margin5_accuracy: .5f} %')
    
    return y_pred_val

check = Test_conv_models(Newconv1_13_model,1700) # VERY BAD MODEL
check = Test_conv_models(NewAttention_13_ResNet1_model,1700)
check = Test_conv_models(Conv1_model_23_model,1700)
check = Test_conv_models(NewResNet_23_model,1700)

other_models(NewRowBlockLSTM_path_model,1700)
# other_models(ResNetCNNweights_13_model,7500)


y_pred = [ np.argmax(VOldResNetCNN_model.predict(np.reshape(x_val_default[idx],(1,-1),order='C'))) for idx in range(1000,1200) ] #len(x_val)
val_exact_accuracy = 100 * sum([1 if y_pred[idx]==y_val[idx]   else 0 for idx in range(len(y_pred)) ]) / len(y_pred)
val_margin5_accuracy = 100 * sum([1 if abs(y_pred[idx]-y_val[idx]) < 5 else 0 for idx in range(len(y_pred)) ]) / len(y_pred)

print(f' Exact accuracy is {val_exact_accuracy: .5f} %')
print(f' Top 5% accuracy is {val_margin5_accuracy: .5f} %')
























        

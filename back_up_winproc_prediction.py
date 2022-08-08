# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:15:21 2022

@author: i368o351
"""



import glob
import numpy as np
from scipy.io import loadmat,savemat
from scipy.signal import convolve2d as conv2
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model,load_model
from IPython.core.interactiveshell import InteractiveShell
from statistics import mode
import random
import csv

from itertools import chain

import os
InteractiveShell.ast_node_interactivity = "all"

# tf.keras.mixed_precision.set_global_policy('mixed_float16')

# PATHS
# Path to data
echo_path = r'Y:\ibikunle\Python_Env\final_layers_rowblock15_21'  # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'
base_echo_path = echo_path + r'\filtered_image' # 'Dec_block_len_45_Test_set191021'  #  < == FIX HERE e.g Full_block_len_45_280921_1530' '\Dec_block_len_21_TEST_set_291021_1519'

# Path to model
#NewAttention_13_ResNet1_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Old_data\Dec_Train_block_len_21_131121_2213\NewResNet_weights_3ConvBlocks\NewResNet_Nov21\NewConv1_model_19_November_21_12_04_Acc_0.6530701518058777_21x15.h5'

model_path1 = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\ResNet_Cnn_input_out_jstars_july21-Copy1\01122022_Acc_0.602_315x1466080.h5'
#model_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\NewAttention_RowBlockLSTM1\11_January_22_16512_Acc_0.781_Top3Acc0.939_21x9.h5' #r'Y:\ibikunle\Python_Env\jstars_weight_21_15_filtered_image_july2021.h5'  #  < == FIX HERE

model_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\PulsedTrainTest\LSTM1_Repeat_NewData\28_January_22_1949_Acc_0.665_Top3Acc0.833_21x9.h5'

prediction_path = r'Y:\ibikunle\Python_Project\Fall_2021\Predictions_Folder'



# Transposed model type?
transposed_LSTM = True

# Confirm path is correct
if os.path.isdir(echo_path):
    print('Target path is okay')
else:
    print('Target path is broken: please fix...')

# Defaults
save_predictions = True
custom_normalize = False
plot_layers = True

#model2 = tf.keras.models.load_model('TBase.h5'); #loaded_model = tf.keras.models.load_model('new_weights_norm65_29_2.h5'); #loaded_model = tf.keras.models.load_model('../jstars_weight_21_15_july2021_resaved.h5')

# Load saved model and weights
loaded_model = tf.keras.models.load_model(model_path)      #model_path,tf.keras.models.load_model('full45x21_echo_Sept2021.h5')

# Load data
#all_echo = glob.glob(echo_path + "/image/*.mat") Usual path

all_echo = glob.glob(echo_path + "//filtered_image//image*.mat") # Just for filtered data
all_echo = sorted(all_echo)    
  
#Load layer e.g all_layer = glob.glob("jstars21_testset/layer/*.mat")
all_layer = glob.glob(echo_path + "/layer/*.mat")
all_layer = sorted(all_layer)  

##      
if len(loaded_model.input_shape) <= 3:
    filt_y = 21; filt_x = 15 #15
else:
    if transposed_LSTM:
        _,filt_x,filt_y,_ = loaded_model.input_shape # (Transposed)       
    else:
        _,filt_y,filt_x,_ = loaded_model.input_shape 
    

conv_filter = np.zeros([filt_y, filt_x])
all_len = filt_y * filt_x


half_x = int((filt_x-1)/2)  
fill_val = filt_y//2

conv_filter[filt_y//2 ,filt_x//2] = 1

if half_x % 2 != 0:
     print('Filter column size is not odd, this might lead to unexpected results')

# Random list of echograms to predict on
#echo_predict_list  = [ random.randint(1,len(all_echo)) for _ in range(5) ] 


test_echo_idx = list(chain.from_iterable( [ list(range(20*n+1,20*n+20)) for n in range(1,1786//20,2) ] ) )    
     
for idx in test_echo_idx[:15]: #: (356,360) #len(all_echo)  echo_predict_list  range(5) range(550,570), range(25,31)
    data = loadmat(all_echo[idx-1]) 
    echo = data['filtered_img'] #data['echo_tmp']
    
    layer_data = loadmat(all_layer[idx-1])
    layer = layer_data['vec_layer'] 
    
    # predictions are the predictions using cumulative pred from model
    # prediction2 uses perfect ground truth knowledge to know where RowBlocks should start
    # predictions with nan replaces all pred==0 with nan (but interpolated values *not* nans are not used to compute the next rowBlock start)
    
    predictions,predictions2 = [],[]
    predictions_w_nan,predictions2_w_nan = [],[]
    raw_prediction = []

    predictions.append(  np.asarray(layer[0,:])  ) # initialize to surface location
    predictions2.append( np.asarray(layer[0,:])  )# initialize to surface location
    
    # Predictions with nan is the version of prediction where all zero prediction is set to NaN
    predictions_w_nan.append(  np.asarray(layer[0,:])  ) # initialize to surface location
    predictions2_w_nan.append( np.asarray(layer[0,:])  )# initialize to surface location
    # raw_prediction
    
    results = {}
    ww = np.where(np.all(np.isnan(layer),1))[0] # Find all rows with nan idx and choose [0] from np.where
    
    if len(ww) == 1:
        num_rows = ww
    else:
        num_rows  = ww [ np.where( np.diff(ww)==1 ) [0][0] ] #First index of all nans in layer  #layer.shape (REMOVE LATER: Uses some knowledge from Ground truth for prediction. This should be avoided)
        
    mod_echo = np.concatenate ( (np.fliplr( echo[:,1:half_x+1]), echo, np.fliplr(echo[:,(-half_x-1):-1] )), axis = 1) # Mirror the edges
    
    if custom_normalize: 
        # Rarely ever use this
        mod_echo = ( mod_echo +65 )/36  # Convert log data back to linear scale
        mod_echo2 = 10**(mod_echo/10)
        mod_echo2 /= np.amax(mod_echo2)
        mod_echo /= 0.0011731572436752015      #np.amax(mod_echo) # normalize    
    
    next_row_block_start = layer[0,:]   
    next_row_block_start2 = layer[0,:]  # This uses given ground truth
    
    Nt,Nx = echo.shape 
    
    # del data; del echo  # Delete some variables because of memory
    all_zero_count = 0
    count = 0
    
    while np.all( np.asarray(next_row_block_start) <= Nt-filt_y ) and all_zero_count < 2 : #
        predict, predict2  = [], []  # re-initialize "predict" for every layer
                     
        for iter_idx in range(Nx): # iterate through 
            row = int(next_row_block_start[iter_idx]) ;
            if not np.all(np.isnan(next_row_block_start2[iter_idx])):
                row2 = int(next_row_block_start2[iter_idx]) # This uses given ground truth
            
            col = int(iter_idx);
            
            if len(loaded_model.input_shape) <= 3:
                predict.append(np.argmax( loaded_model.predict( (conv2( conv_filter, mod_echo [row:row+filt_y, col:col+filt_x], mode='same').T.ravel()).reshape(1,all_len))  )  )             
                predict2.append(np.argmax( loaded_model.predict( (conv2( conv_filter, mod_echo [row2:row2+filt_y, col:col+filt_x], mode='same').T.ravel()).reshape(1,all_len))  )  ) 
                
            else:
                
                if transposed_LSTM:
                    curr_data = np.expand_dims ( np.transpose(mod_echo [row:row+filt_y, col:col+filt_x]),axis= 0 )
                    curr_data2 = np.expand_dims ( np.transpose(mod_echo[row2:row2+filt_y, col:col+filt_x]), axis= 0)
                    predict.append(np.argmax(loaded_model.predict(curr_data)))             
                    predict2.append(np.argmax( loaded_model.predict( curr_data2 )) )                     
                
                else:                               
                    curr_data = np.expand_dims ( mod_echo [row:row+filt_y, col:col+filt_x],axis= 0)
                    curr_data2 = np.expand_dims (mod_echo [row2:row2+filt_y, col:col+filt_x],axis= 0)
                    predict.append(np.argmax(loaded_model.predict(curr_data)))             
                    predict2.append(np.argmax( loaded_model.predict( curr_data2 )) ) 
        
        # Correct "over-zealous" prediction
        if not np.all ( np.asarray(predict) == 0):
            prediction_mode_val = mode(np.array(predict)[ np.array(predict) > 0] ) + half_x
            
            if np.any(  abs(np.asarray(predict) - prediction_mode_val) > filt_y//2 ):
                predict = np.asarray(predict)
                
                # Only remove if there are only few (<5) instances of such predictions
                if sum(abs(np.asarray(predict) - prediction_mode_val) > filt_y//2) > 5:
                    predict[abs(np.asarray(predict) - prediction_mode_val) > filt_y//2 ] = 0 
             

        if np.any ( np.asarray(predict) == 0):
            
            # Interpolate to fill missing zeros
            ## Check if all prediction is zero ##
            if np.all( np.asarray(predict) == 0 ):
              all_zero_count +=1
              past_pred_zeros = True
              predict_intpd = fill_val
            
            else:                
              # Just some of the predictions have zeros 
              predict_intpd = pd.Series(predict)
              predict_intpd [predict_intpd == 0] = np.nan
              x= np.arange(predict_intpd.size)
              predict_intpd[np.isnan(predict_intpd)] = ( np.interp(x[np.isnan(predict_intpd)], x[np.isfinite(predict_intpd)],predict_intpd[np.isfinite(predict_intpd)])  ).astype('int')
              past_pred_zeros = False
              predict_intpd = predict_intpd.to_numpy()
              
        else:
            # No zeros in the predictions
          predict_intpd = predict
          past_pred_zeros = False          

        
        ## Determine the next rowBlock start  
        next_row_block_start = next_row_block_start + predict_intpd        
        
        count +=1
        if count < num_rows-1:            
            next_row_block_start2 = layer[count,:] 
        
        # if count > num_rows + 1:
        #     break
        
        
        ## The predictions by the model for a layer may contain 0, those predictions are set to NaN and saved: predictions_w_nan is the real prediction 
        # However, for calculating the next rowBlock start, missing gaps should be interpolated
        temp = next_row_block_start.copy()
        
        if any( np.asarray(predict)==0  ):
            temp[ np.asarray(predict)==0 ] = np.nan      
        
        
        ## Cumulate predictions
        #============================================       

        predictions.append(next_row_block_start) # Interpolated and perfect prediction
        predictions2.append(predict2 + predictions2[-1] ) # Prediction using perfect ground truth           
        predictions_w_nan.append(temp + predictions[-1]) #Prediction with nans for predicted zeros        
        raw_prediction.append(predict) # raw predictions for each layer      
          
    if plot_layers:        
        fig, axes = plt.subplots(figsize=(15,30),dpi = 100);
        _ = axes.imshow(echo,cmap='gray_r');
        _ = axes.plot( np.arange(Nx),np.asarray(layer).T,'b-', np.arange(Nx),np.asarray(predictions).T,'g--' );
        
#============================================
## Cumulate prediction and save predictions
#============================================
        
    fn ='echo_prediction'       
    results[fn] = predictions
    
    fn = 'gt_prediction'   # Predictions using perfect ground truth    
    results[fn] = predictions2#
    
    fn = 'prediction_w_nan'        
    results[fn] = predictions_w_nan
    
    # fn = 'raw_prediction'
    # raw_prediction = np.insert( raw_prediction, 0, layer[0] )        
    # results[fn] = predictions_w_nan
    
    if save_predictions:        

 
        fn = '/predictions_' +  os.path.basename(all_layer[idx]) # 'newest_predictions_echo%06d.mat'% (idx+1)  #os.path.splitext        
        if not os.path.isdir( prediction_path+ f'/Predictions_Folder_{filt_y}_{filt_x}' ):
            os.makedirs( prediction_path+f'/Predictions_Folder_{filt_y}_{filt_x}' )
    
        save_path = prediction_path + f'/Predictions_Folder_{filt_y}_{filt_x}' + fn      
           
        savemat(save_path, results)
        
        if not os.path.isfile( prediction_path+ f'/Predictions_Folder_{filt_y}_{filt_x}/ReadMe.csv' ):
            
            field_names = ['model_path', 'echo_path']
            ReadMeContent = {'model_path': model_path,
                             'echo_path':echo_path
                            }
                
            with open('ReadMe.csv', 'w') as csvfile:          
                
                writer = csv.DictWriter(csvfile, fieldnames = field_names)
                writer.writeheader()
                writer.writerows(ReadMeContent)
##    
##    del predictions,predictions2,results
##    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 01:33:41 2022
% Generate and plot metrics 

@author: i368o351
"""


import tensorflow as tf
import glob
import os
from scipy.io import loadmat,savemat
import numpy as np
from matplotlib import pyplot as plt
import random
from keras import backend as K # This is needed by AttentionUNet

from scipy.ndimage import median_filter as sc_med_filt

from sklearn.metrics import (accuracy_score,recall_score,f1_score,cohen_kappa_score, 
            classification_report,PrecisionRecallDisplay,precision_recall_curve, precision_score)

import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()


#==============================================================================
# Paths
#==============================================================================
base_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data'  # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'
test_path = os.path.join(base_path,'test_data\*.mat') 

output_base_path = r'Y:\ibikunle\Python_Project\Fall_2021\Predictions_Folder\DL_models_predictions_folder_final'


# =============================================================================
## Function for test and validation dataset    
# =============================================================================

def read_mat_files(data_fp,prediction_fp):
    all_raster = []
    all_model_binarized = []
    all_model_out = []
    
    for file in glob.glob(data_fp):
        base_name = os.path.basename(file)        
        output_file = os.path.join(prediction_fp,base_name)
        
        temp_test_file = loadmat(file)
        temp_output = loadmat(output_file)
        
        curr_GT_raster = np.asarray(temp_test_file['raster'],dtype=bool).astype('int32')
        curr_binary_out = np.asarray(temp_output['binary_output'],dtype=bool).astype('int32')
        
        all_raster.append(curr_GT_raster)
        all_model_binarized.append(curr_binary_out)
        all_model_out.append(temp_output['model_output'])
    return all_raster,all_model_binarized, all_model_out

# =============================================================================
## Function to create metrics for each model output
# =============================================================================
model_folders = ['all_AttUNet', 'all_DeepLab','all_ensemble','all_FCN','all_SimpleUNet']  


all_metrics = {}

for elem in model_folders:
    all_metrics[elem] = {}
    
    curr_output_path = os.path.join(output_base_path,elem)
    
    GT_raster, model_raster, model_output =  read_mat_files(test_path,curr_output_path) 
    
    GT_raster = np.array(GT_raster,dtype='int32').flatten()
    model_raster = np.array(model_raster,dtype='int32').flatten()
    
    ## Metrics
    # 1. Average Accuracy (AA) (average of recalls)       
    recall = recall_score(GT_raster, model_raster, average='weighted')   
    all_metrics[elem]['recall'] = recall
    
    # 1b. Precision      
    precision = precision_score(GT_raster, model_raster, average='weighted')   
    all_metrics[elem]['precision'] = precision
    
    # 2. Overall Accuracy (OA)      
    acc = accuracy_score(GT_raster, model_raster)
    all_metrics[elem]['acc'] = acc    
    
    # 3. F1 score
    f1 = f1_score(GT_raster, model_raster, average='weighted')
    all_metrics[elem]['f1'] = f1
    
    # 4. Kappa
    Kp = cohen_kappa_score(GT_raster, model_raster)
    all_metrics[elem]['Kp'] = Kp    
    
    # 5. Classification report
    report = classification_report(GT_raster, model_raster,output_dict=True)
    all_metrics[elem]['report'] = report    
    
    # 6. Precision_Recall_curve
    # disp = PrecisionRecallDisplay.from_predictions(GT_raster, model_raster, ax = plt.gca(), name = elem )
   
    disp = PrecisionRecallDisplay(report['weighted avg']['precision'], report['weighted avg']['recall']) #, ax = plt.gca(), name = elem )
    disp.plot(ax =plt.gca(), name=elem)

## Save result
out_path = os.path.join(output_base_path,'extended_metrics.mat')
savemat(out_path,all_metrics)

plt.show()








































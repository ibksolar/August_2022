# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:12:30 2022
Compare model outputs
@author: i368o351
"""

from tensorflow.keras import layers
import tensorflow as tf
import glob
import os
from scipy.io import loadmat,savemat
import numpy as np
from matplotlib import pyplot as plt

import matplotlib as mpl
import cv2

import random
from keras import backend as K # This is needed by AttentionUNet
import scipy
from scipy.ndimage import median_filter as sc_med_filt

from statistics import mode

import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()

from custom_binarize import custom_binarize
from create_vec_layer import create_vec_layer

from skimage.morphology import rectangle
from skimage.filters.rank import modal

from matplotlib import colors



# Set the default color cycle
ab = list( mpl.cycler(mpl.rcParams['axes.prop_cycle']) )
clr = [ item['color'] for item in ab if item['color'] != '#7f7f7f' ] # Remove illegible color
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=clr) 


#==============================================================================
# PATHS
#==============================================================================
base_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data'  # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'
# test_path = os.path.join(base_path,'grouped_test_data\*.mat')  
# test_data = glob.glob(test_path)

L1_files = glob.glob(os.path.join(base_path,r'grouped_test_data\L1\*.mat') )
L2_files = glob.glob(os.path.join(base_path,r'grouped_test_data\L2\*.mat') )
L3_files = glob.glob(os.path.join(base_path,r'grouped_test_data\L3\*.mat') )

test_data = L1_files + L2_files + L3_files


#out_dir = os.path.join(base_path,'predictions_folder')
out_dir = r'Y:\ibikunle\Python_Project\Fall_2021\Predictions_Folder\DL_models_predictions_folder_final2'

#==============================================================================
# PatchEncoder and others
#==============================================================================

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches,  embed_dim,  **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=embed_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=embed_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    # # Override function to avoid error while saving model
    # def get_config(self):
    #     config = super().get_config().copy()
    #     config.update(
    #         {               
    #             "num_patches": num_patches,
    #         }
    #     )
    #     return config

#==============================================================================    
#==============================================================================


#==============================================================================
# Visualize model prediction
#==============================================================================


def viz_model_pred(model,model_name, 
                   batch_idx = random.randint(0,len(test_data)-10), # Get a random batch
                   rmv_island = False, # Remove island predictions
                   save_pred = False, # Save only a few predictions
                   save_all = False, # Save ALL predictions in the test set
                   save_plot = False ):
    
    used_range = range(0,len(test_data)) if save_all else range(batch_idx,batch_idx+10)
    
    for idx in used_range:
       # print(f'{os.path.basename(test_data[idx])}')  
      if  1: #not os.path.basename(test_data[idx]) != '20120330_04_0324_5km.mat': #or os.path.basename(test_data[idx]) != '20120330_04_1118_2km.mat'
          predict_data = loadmat(test_data[idx])
          L_folder = test_data[idx].split(os.path.sep)[-2]
          base_name = os.path.basename(test_data[idx])
          print(f'{base_name}')
          
          # a01,a_gt0 = predict_data['echo_tmp'], predict_data['raster']
          a01 = predict_data['echo_tmp']
          Nt,Nx = a01.shape
          stop_val = np.argmax( np.diff( np.sum(a01,1) == 0) ) if np.argmax( np.diff( np.sum(a01,1) == 0) ) > 0.5*Nt else Nt
          
          # Constant params
          min_rows = 20
          filter_x,filter_y = 5, 15    #int(Nt//80), int(Nx//8)
          conv_filter = np.ones(shape=(filter_x,filter_y))/filter_y
          kernel = np.ones((3,7), np.uint8)
          
          
          
          if type(model) == list:
              res0 = sum( [ _load_predict(m1,predict_data) for m1 in model])/len(model)
          else:
              if model.input_shape[-1]  >1:
                  a0 = np.stack((a01,)*3,axis=-1)
                  res0 = model.predict ( np.expand_dims(a0,axis=0))
              else:
                  res0 = model.predict ( np.expand_dims(np.expand_dims(a01,axis=0),axis=3) )       
              res0 = res0 if len(res0)==1 else res0[0]              
          
          res0 = res0.squeeze()
          
          # Dilate model output
          res0_dil = cv2.dilate(res0, kernel, iterations=1)         
     
          # Filter
          res0 = scipy.signal.convolve2d(res0,conv_filter,mode='same',boundary='symm') 
          
          # C2 = res0.copy()
          # C2 [np.isnan(C2)] = 0
          # counts,bins = np.histogram( C2.ravel() )              
          # binarize_threshold =  np.max( (bins[ np.argmax(counts) +1 ], bins[ np.argmax(counts) +2 ], np.percentile(res0,80)) )
          
          # binarize_threshold = md + 0.12*(np.max(res0) - md) if np.sum(res0>md)/np.prod(res0.shape) > 0.2 else np.percentile(res0,75)
          binarize_threshold =  np.percentile(res0,75)
          
          res0_final = np.where(res0_dil>binarize_threshold,1,0)          
          
          res0_final[:min_rows,:] = 0 # Set all prediction before 20th row to zero
          
          # Better to truncate after thresholding to accomodate negative prob value models
          res0[stop_val:,: ] = 0          
          res0_final[stop_val:,: ] = 0            

          if rmv_island: # Usually not used
          # Remove islands and discontinuities in thresholded predictions          
              filter_x,filter_y = int(Nt//80), int(Nx//5)
              conv_filter = np.ones(shape=(filter_x,filter_y ))              
              res0_island_rmv = res0_final.copy()               
              # Filter top
              # conv_vals = cv.filter2D(res0_island_rmv, -1, conv_filter, borderType=cv.BORDER_CONSTANT)  
              conv_vals = scipy.signal.convolve2d(res0_island_rmv,conv_filter,mode='same',boundary='symm') 
              binarize_threshold2 = np.percentile(conv_vals,65)              
              res0_island_rmv_top = res0_island_rmv[:int(.75*Nt),:]
              conv_vals = conv_vals[:int(.75*Nt),:]              
              res0_island_rmv_top[conv_vals <binarize_threshold2 ] = 0 #np.max(conv_vals)//5              
              # Filter bottom              
              res0_island_rmv = modal(res0_island_rmv.astype(np.uint8), rectangle(7,17) ) # Majority filter (Larger kernels more aggressive)           
              
          
          res0_new = res0_island_rmv if rmv_island else res0_final # This is the new binarized map
          
          cbin = custom_binarize(res0,res0_new, closeness = 15, return_segment = False)
          
          res0_final1 = np.arange(1,Nt+1).reshape(Nt,1) *  cbin 
          
         
          thresh = 12 #{'constant': 30}
          vec_layer = create_vec_layer(res0_final1,thresh); 
        
          
          vec_layer[vec_layer==0] = np.nan
          vec_layer[vec_layer >= stop_val] = np.nan
          
          new_layer_filtered = vec_layer.copy()
          new_layer_filtered[:] = np.nan  
          
          for chan in range(new_layer_filtered.shape[0]):
                new_layer_curr = vec_layer[chan,:]
                if ~np.all(np.isnan(new_layer_curr)) and len(new_layer_curr[~np.isnan(new_layer_curr)]) > 21:
                    new_layer_filtered[chan,:] =  sc_med_filt(new_layer_curr, size=55).astype('int32') #sc_med_filt(z,size=3)
                else:
                    new_layer_filtered[chan,:] = np.nan
          new_layer_filtered [ new_layer_filtered< 0] = np.nan 
          del_idx = np.argwhere(np.sum(new_layer_filtered,axis=1)==0) # Find "all zero" rows              
          new_layer_filtered = np.delete(new_layer_filtered,del_idx,axis = 0) # Delete them
          
          new_layer_filtered [new_layer_filtered==0] = np.nan
          short_layers = np.argwhere( np.sum(np.isnan(new_layer_filtered),axis = 1) > Nx//1.3)
          
          # incomp_wavy_layers = np.argwhere(np.nansum( np.abs(np.diff(new_layer_filtered,axis = 1)),axis=1) > 100 ) & (np.argwhere( np.sum(~np.isnan(new_layer_filtered),axis=1) < round(0.5*Nx))).T #np.nansum( np.abs(np.diff(new_layer_filtered,axis = 1))
          # short_layers = np.append( short_layers, incomp_wavy_layers ) 
          
          new_layer_filtered = np.delete(new_layer_filtered,short_layers,axis = 0) 
          
          if save_pred or save_all:
              
              if not os.path.exists(os.path.join(out_dir,L_folder, model_name)):
                  os.makedirs(os.path.join(out_dir,L_folder,model_name), exist_ok=True) 
                              
              save_path = os.path.join(out_dir,L_folder,model_name,base_name)
              out_dict = {} 
              out_dict['model_output'] = res0
              out_dict['binary_output'] = res0_final1
              out_dict['vec_layer'] = vec_layer
              out_dict['filtered_vec_layer'] = new_layer_filtered
              out_dict['GT_layer'] = predict_data['vec_layer']
              savemat(save_path,out_dict)
              
              
          if save_plot:
              
              f, axarr = plt.subplots(1,7,figsize=(20,20))
            
              axarr[0].imshow(a01.squeeze(),cmap='gray_r')
              axarr[0].set_title( f'Echo {os.path.basename(test_data[idx])}') #.set_text
              
              axarr[1].imshow(a01,cmap='viridis', norm=colors.SymLogNorm(linthresh=0.03, vmin=.85*a01[a01>0].min(), vmax=a01.max()) )
              axarr[1].set_title('Orig echo map')
              
              axarr[2].imshow(res0, cmap='viridis', norm=colors.SymLogNorm(linthresh=0.03, vmin=.85*res0[res0>0].min(), vmax=res0.max()) )
              axarr[2].set_title(f'{model_name}_output before threshold') 
              
              axarr[3].imshow(res0_final1.astype(bool).astype(int), cmap='viridis' )
              axarr[3].set_title(f'{model_name}_Prediction') 
              
              # axarr[4].imshow(seg_map, cmap='viridis' )
              # axarr[4].set_title(f'{model_name}_Seg_map')              
            
              axarr[4].imshow(a01,cmap='gray_r')
              axarr[4].plot(vec_layer.T) # gt    
              axarr[4].set_title( f'Vec_layer({thresh} overlaid)') #.set_text
              
              axarr[5].imshow(a01.squeeze(),cmap='gray_r')          
              axarr[5].plot(new_layer_filtered.T) # gt
              axarr[5].set_title( 'Filtered Overlaid prediction') #.set_text
              
              axarr[6].imshow(a01.squeeze(),cmap='gray_r')          
              axarr[6].plot(predict_data['vec_layer'].T) # gt
              axarr[6].set_title( 'Overlaid GT') #.set_text

              
              base_name = os.path.basename(test_data[idx])                  
              if not os.path.exists(os.path.join(out_dir,L_folder,model_name)):
                 os.makedirs(os.path.join(out_dir,L_folder,model_name),exist_ok=True) 
              if not os.path.exists(os.path.join(out_dir,L_folder,model_name,'plotted_images')):
                 os.makedirs(os.path.join(out_dir,L_folder,model_name,'plotted_images'),exist_ok=True) 
            
              save_fig_path = os.path.join(out_dir,L_folder,model_name,'plotted_images',base_name)
              save_fig_path,_ =  os.path.splitext(save_fig_path)
              plt.savefig(save_fig_path+'.png')        
              f.clf()
              plt.close()
          
def _load_predict(model,predict_data):
      a01 = predict_data['echo_tmp']
      Nt,Nx = a01.shape
      
      if model.input_shape[-1]  >1:
          a0 = np.stack((a01,)*3,axis=-1)
          res0 = model.predict ( np.expand_dims(a0,axis=0))
      else:
          res0 = model.predict ( np.expand_dims(np.expand_dims(a01,axis=0),axis=3) ) 
      
      res0 = res0 if len(res0)==1 else res0[0]
      res0 = res0.squeeze()
      
      return res0       

#==============================================================================
# CUSTOM OBJECT
#==============================================================================
custom_objects ={'PatchEncoder':PatchEncoder, 'binary_focal_loss':sm.losses.BinaryFocalLoss(), 'iou_score':sm.metrics.iou_score, 'precision':sm.metrics.precision} 

#==============================================================================
# LOAD MODELS
#==============================================================================
batch_idx = 168 #random.randint(0,len(test_data)-10)




# (0.) FCN (Not good atm: Really GOOD now!)
FCN_path = os.path.join(base_path,r'SimpleFCNet\SimpleFCNet_Checkpoint22_November_22_2221.h5')
FCN = tf.keras.models.load_model(FCN_path, custom_objects=custom_objects)
# viz_model_pred(FCN,'FCN',batch_idx, save_pred=True, save_plot=True)
# viz_model_pred(FCN,'FCN',batch_idx, save_all=True, save_plot=True)

# (1.) SimpleUNet
SimpleUNet_path = os.path.join(base_path,r'SimpleUNet_Binary_large\SimpleUNet_acc_0.99_no_fixed_shape_13_October_22_0805.h5')
SimpleUNet = tf.keras.models.load_model(SimpleUNet_path)
viz_model_pred(SimpleUNet,'SimpleUNet',batch_idx)
# viz_model_pred(SimpleUNet,'SimpleUNet',batch_idx, save_pred=True, save_plot=True)
# viz_model_pred(SimpleUNet,'SimpleUNet',batch_idx, save_all=True, save_plot=True)

# (2.) AttentionUNet
AttUNet_path = os.path.join(base_path,r'AttUNet\AttUNet_Checkpoint14_October_22_0940.h5')
AttUNet = tf.keras.models.load_model(AttUNet_path, custom_objects = custom_objects)
# viz_model_pred(AttUNet,'AttUNet',batch_idx, save_pred=True, save_plot=True)
# viz_model_pred(AttUNet,'AttUNet',batch_idx, save_all=True, save_plot=True)

# (3.) DeepLab
DeepLab_path = os.path.join(base_path,r'DeepLab_binary_large\DeepLab_Checkpoint17_November_22_1149.h5')
DeepLab = tf.keras.models.load_model(DeepLab_path, custom_objects = custom_objects)
# viz_model_pred(DeepLab,'DeepLab',batch_idx, save_pred=True, save_plot=True)
# viz_model_pred(DeepLab,'DeepLab',batch_idx, save_all=True, save_plot=True)
  
# (4.) Multiple output Res50_pretrained
Mult_pretrained_path = os.path.join(base_path,'pretraining_model\pretraining_model_Checkpoint14_October_22_0555.h5')
mult_pretrained = tf.keras.models.load_model(Mult_pretrained_path, custom_objects = custom_objects)
# viz_model_pred(mult_pretrained,'mult_pretrained',batch_idx, save_pred=True, save_plot=True)
# viz_model_pred(mult_pretrained,'mult_pretrained',batch_idx, save_all=True, save_plot=True)
# # (4b.) pretrained
# pretrained_path = os.path.join(base_path,'pretraining_model\pretraining_model_Checkpoint14_October_22_0555.h5')
# pretrained = tf.keras.models.load_model(pretrained_path, custom_objects = custom_objects)
# viz_model_pred(pretrained,'pretrained',batch_idx)

# (5.) Res50_pretrained ( This can take any input shape)
Res50_pretrained_path = os.path.join(base_path,'Res50_pretrained_model\Res50_pretrained_model_98.51_02_November_22_1845.h5')
Res50_pretrained = tf.keras.models.load_model(Res50_pretrained_path, custom_objects = custom_objects)
# viz_model_pred(Res50_pretrained,'Res50_pretrained',batch_idx , save_pred=True, save_plot=True)
# viz_model_pred(Res50_pretrained,'Res50_pretrained',batch_idx , save_all=True, save_plot=True)

# (6) Ensemble
models = [ FCN, SimpleUNet, AttUNet,  Res50_pretrained ] #DeepLab,
viz_model_pred(models,'ensemble',batch_idx , save_all=True, save_plot=True)











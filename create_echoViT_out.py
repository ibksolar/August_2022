# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 23:26:42 2023

@author: i368o351
"""


import tensorflow as tf
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm,colors
from statistics import mode
import os
from scipy import signal

import cv2

from scipy.io import loadmat,savemat
from scipy.ndimage import median_filter as sc_med_filt

from custom_binarize import custom_binarize
from create_vec_layer import create_vec_layer

# from keras.metrics import MeanIoU
# from sklearn.metrics import roc_auc_score
from skimage.morphology import remove_small_objects
import glob



# Model
model_path = r'U:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data\EchoViT_paper\EchoViT_paper_OKAY_binary_30_December_22_2204.h5'
model = tf.keras.models.load_model(model_path)


# PATHS
# Path to data
base_path = r'U:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data'  

output_names = ['rw_embed_out', 'col_embed_out', 'patch_embed_out', 'ResNet_embed_out', 'combined_out']


save_pred = 1
save_fig = 1

# Set the default color cycle
ab = list( mpl.cycler(mpl.rcParams['axes.prop_cycle']) )
clr = [ item['color'] for item in ab if item['color'] != '#7f7f7f' ] # Remove illegible color
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=clr) 


L1_files = glob.glob(os.path.join(base_path,r'test_L_files\L1\*.mat') )
L2_files = glob.glob(os.path.join(base_path,r'test_L_files\L2\*.mat') )
L3_files = glob.glob(os.path.join(base_path,r'test_L_files\L3\*.mat') )
test_data =  L1_files + L2_files + L3_files #

#out_dir = os.path.join(base_path,'predictions_folder')
out_dir = r'M:\ibikunle\Python_Project\Fall_2021\Predictions_Folder\EchoViT_out2\filtered_version'

L_files = [L1_files,L2_files,L3_files] # L1_files,
L_folders = ['L1','L2','L3'] # 'L1',

kernel = np.ones((11,35), np.uint8)
erode_kernel = np.ones((3,5), np.uint8)



for idx0 in range(len(L_files)):
  for curr_data in L_files[idx0]:
      if '20120330_03_0006_10km'  in curr_data : #1: #'20120330_04_0952_2km', '20120330_04_0378_5km'  in curr_data: 'L1' not in curr_data
          predict_data = loadmat(curr_data)
          a01,a_gt0 = predict_data['echo_tmp'], predict_data['raster']
          
          base_name = os.path.basename(curr_data)
          L_folder = L_folders[idx0]
          
          print(f' File {base_name} in {L_folder} ')
          
          Nt,Nx = a01.shape
          stop_val = np.argmax( np.diff( np.sum(a01,1) == 0) ) if np.argmax( np.diff( np.sum(a01,1) == 0) ) > 0.15*Nt else Nt #np.any( np.sum(a01[int(.6*Nt):,:],1) == 0)
          
          if model.input_shape[-1] > 1:
            a0 = np.stack((a01,)*3,axis=-1)
            res0_all = model.predict ( np.expand_dims(a0,axis=0))
          else:
            res0_all = model.predict ( np.expand_dims(np.expand_dims(a01,axis=0),axis=3) )   
         
          for mod_idx in range(len(res0_all)):
              res0 = res0_all[mod_idx].squeeze() #max ( [ res0_all[mod_idx].squeeze().tolist()  for mod_idx in [0,1,3] ] )             
              
              model_name = output_names[mod_idx]
              
              res0 = np.asarray(res0)
              
              # Erode res0 to remove tiny artifacts
              # res0 = cv2.erode(res0, erode_kernel)            
              
              # Filter model output              
              filter_x,filter_y = 21, 75     #int(Nt//80), int(Nx//8)
              conv_filter = np.ones(shape=(filter_x,filter_y))/filter_y          
              res02 = signal.convolve2d(res0,conv_filter,mode='same',boundary='symm')   # Add res0 to compensate for not flat edges          
              res02 = res02 + res0  # 2*cv2.bitwise_and(res0,res02)
              
              # Get threshold
              mod_val = mode(res02.ravel())
              binarize_threshold =  np.percentile(res0,79) #mod_val + 0.02*(np.max(res02) - mod_val) #np.percentile(res0,97)
              
              # Dilate model output
              res0_dil = cv2.dilate(res0, kernel, iterations=1) #if 'L1' in L_folder else cv2.dilate(res0, kernel, iterations=2)
              res0_thresholded_where = np.where(res0_dil>binarize_threshold,1,0) # DON'T USE DILATE HERE, use filtered instead or maybe just res0
              
              
              # (3.) Remove above and below artifacts
              res0_thresholded_where[:20,:] = 0 # Set all prediction before 20th row to zero          
              # Better to truncate after thresholding to accomodate negative prob value models
              res0_thresholded_where[stop_val:,: ] = 0          
               
              cbin = custom_binarize(res0,res0_thresholded_where, closeness = 10, return_segment = False)  
              
              cbin_rmv = remove_small_objects(cbin.astype('bool'),min_size= 10, connectivity=8)
              res0_binarized_thin_rbin = np.arange(1,Nt+1).reshape(Nt,1) *  cbin_rmv  
              
              res0_final_mod =  res0_binarized_thin_rbin.copy()
              
              
              # How correct is create_vec_layer??
              # thresh = {'constant': 25}
              thresh = 12
              vec_layer = create_vec_layer( res0_binarized_thin_rbin,thresh);      
              vec_layer[vec_layer==0] = np.nan
            
              
              new_layer_filtered = vec_layer.copy()
              new_layer_filtered[:] = np.nan      
              for chan in range(new_layer_filtered.shape[0]):
                    new_layer_curr = vec_layer[chan,:]
                    if ~np.all(np.isnan(new_layer_curr)) and len(new_layer_curr[~np.isnan(new_layer_curr)]) > 21:
                        new_layer_filtered[chan,:] =  np.round( sc_med_filt(new_layer_curr, size=55) ) #sc_med_filt(z,size=3)
                    else:
                        new_layer_filtered[chan,:] = np.nan
              new_layer_filtered [ new_layer_filtered< 0] = np.nan 
              del_idx = np.argwhere(np.sum(new_layer_filtered,axis=1)==0) # Find "all zero" rows              
              new_layer_filtered = np.delete(new_layer_filtered,del_idx,axis = 0) # Delete them
              
              new_layer_filtered [new_layer_filtered==0] = np.nan
              short_layers = np.argwhere( np.sum(np.isnan(new_layer_filtered),axis = 1) > round(.75*Nx) )              
              # incomp_wavy_layers = np.argwhere(np.nansum( np.abs(np.diff(new_layer_filtered,axis = 1)),axis=1) > 70 ) & (np.argwhere( np.sum(~np.isnan(new_layer_filtered),axis=1) < round(0.5*Nx))).T #np.nansum( np.abs(np.diff(new_layer_filtered,axis = 1))
              # short_layers = np.append( short_layers, incomp_wavy_layers ) 
              
              new_layer_filtered = np.delete(new_layer_filtered,short_layers,axis = 0) 
        
              pred0_final = sc_med_filt( sc_med_filt(res0_thresholded_where,size=7).T, size=7, mode='nearest').T
              
              if save_fig:
                  if not os.path.exists(os.path.join(out_dir,L_folder, model_name,'plotted_images')):
                      os.makedirs(os.path.join(out_dir,L_folder,model_name,'plotted_images'), exist_ok=True)        
            
                  f, axarr = plt.subplots(1,8,figsize=(20,20))
                      
                  axarr[0].imshow(a01.squeeze(),cmap='gray_r')
                  axarr[0].set_title( f'Echo {os.path.basename(curr_data)}') #.set_text
                    
                  axarr[1].imshow(res02,cmap='viridis' ) #, norm=colors.SymLogNorm(linthresh=0.03, vmin=.85*a01.min(), vmax=a01.max())
                  axarr[1].set_title('Regions')
                  
                  axarr[2].imshow(res0_thresholded_where,cmap='viridis' ) #, norm=colors.SymLogNorm(linthresh=0.03, vmin=.85*a01.min(), vmax=a01.max())
                  axarr[2].set_title('Regions')
                    
                  axarr[3].imshow(cbin, cmap='viridis', norm=colors.SymLogNorm(linthresh=0.03, vmin=.85*res0.min(), vmax=res0.max()) )
                  axarr[3].set_title(f'{model_name}_output') 
                    
                  axarr[4].imshow(res0_binarized_thin_rbin.astype(bool).astype(int), cmap='viridis' )
                  axarr[4].set_title(f'{model_name}_prediction') 
                    
                  axarr[5].imshow(a01,cmap='gray_r')
                  axarr[5].plot(vec_layer.T) # gt    
                  axarr[5].set_title( f'Vec_layer({thresh} overlaid)') #.set_text
                    
                  axarr[6].imshow(a01.squeeze(),cmap='gray_r')          
                  axarr[6].plot(new_layer_filtered.T) # gt
                  axarr[6].set_title( 'Filtered Overlaid prediction') #.set_text
                    
                  axarr[7].imshow(a01.squeeze(),cmap='gray_r')          
                  axarr[7].plot(predict_data['vec_layer'].T) # gt
                  axarr[7].set_title( 'Overlaid GT') #.set_text             
    
                  
                  save_fig_path = os.path.join(out_dir,L_folder,model_name,'plotted_images',base_name)
                  save_fig_path,_ =  os.path.splitext(save_fig_path)
                  plt.savefig(save_fig_path+'.png')        
                  
                  plt.close()
              
              
                    
              if save_pred:
                  if not os.path.exists(os.path.join(out_dir,L_folder, model_name)):
                      os.makedirs(os.path.join(out_dir,L_folder,model_name), exist_ok=True) 
                                
                  save_path = os.path.join(out_dir,L_folder,model_name,base_name)
                  out_dict = {} 
                  out_dict['model_output'] = res0
                  out_dict['binary_output'] =  res0_binarized_thin_rbin
                  out_dict['vec_layer'] = vec_layer
                  out_dict['filtered_vec_layer'] = new_layer_filtered
                  out_dict['GT_layer'] = predict_data['vec_layer']
                  savemat(save_path,out_dict)

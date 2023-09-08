# -*- coding: utf-8 -*-
"""
Created on Sat May 27 10:56:43 2023

@author: i368o351
"""

from create_vec_layer_updated import create_vec_layer2

from skimage.morphology import rectangle,remove_small_objects, remove_small_holes,dilation, area_opening
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
from create_vec_layer_updated import create_vec_layer
from skimage.morphology import rectangle,remove_small_objects, remove_small_holes,dilation,binary_closing
# from keras.metrics import MeanIoU
# from sklearn.metrics import roc_auc_score

import glob



# Model
model_path = r'M:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data\EchoViT_paper\EchoViT_paper_OKAY_binary_30_December_22_2204.h5'
model = tf.keras.models.load_model(model_path)


# PATHS
# Path to data
base_path = r'M:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data'  

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

model_names = ['rw_embed_out', 'col_embed_out', 'patch_embed_out', 'ResNet_embed_out', 'combined_out' ]

for idx0 in range(len(L_files)):
  for curr_data in L_files[idx0]:
      if 1: #'20120330_04_0843_2km'  in curr_data : #1: #'20120330_04_0952_2km', '20120330_04_0378_5km'  in curr_data: 'L1' not in curr_data
          predict_data = loadmat(curr_data)
          a01,a_gt0 = predict_data['echo_tmp'], predict_data['raster']
          
          base_name = os.path.basename(curr_data)
          L_folder = L_folders[idx0]
          
          print(f' File {base_name} in {L_folder} ')
          
          Nt,Nx = a01.shape
          stop_val = np.argmax( np.diff( np.sum(a01,1) == 0) ) if np.argmax( np.diff( np.sum(a01,1) == 0) ) > 0.5*Nt else Nt
          
          if model.input_shape[-1] > 1:
            a0 = np.stack((a01,)*3,axis=-1)
            res0_all = model.predict ( np.expand_dims(a0,axis=0))
          else:
            res0_all = model.predict ( np.expand_dims(np.expand_dims(a01,axis=0),axis=3) ) 

         
          for mod_idx in [0,1,3]: # range(len(res0_all)):
              res0 = res0_all[mod_idx].squeeze() #max ( [ res0_all[mod_idx].squeeze().tolist()  for mod_idx in [0,1,3] ] )             
              
              model_name = output_names[mod_idx]
              
              res0 = np.asarray(res0)
              
              # Erode res0 to remove tiny artifacts
              # res0 = cv2.erode(res0, erode_kernel)            
              
              # Filter model output              
              filter_x,filter_y = 7, 7  # Seems like shorter filter helps. Verfiying 28th May,2023   #int(Nt//80), int(Nx//8)
              conv_filter = np.ones(shape=(filter_x,filter_y))         
              res02 = 1.5*res0 + signal.convolve2d(res0,conv_filter,mode='same',boundary='symm')   # Add res0 to compensate for not flat edges          
              # res02 = res02 + res0  # 2*cv2.bitwise_and(res0,res02)
              res02 = signal.convolve2d(res02,conv_filter,mode='same',boundary='symm')
              
              # Get threshold
              mod_val = mode(res02.ravel())
              binarize_threshold =  np.percentile(res0,87) #mod_val + 0.02*(np.max(res02) - mod_val) #np.percentile(res0,97)
              
              # Dilate model output
              res0_dil = cv2.dilate(res0, kernel, iterations=1) #if 'L1' in L_folder else cv2.dilate(res0, kernel, iterations=2)
              res0_thresholded_where = np.where(res0>binarize_threshold,1,0)
              
              # Close in-between gaps 
              res0_thresholded_where = binary_closing(res0_thresholded_where.astype('bool'),footprint = np.ones((5,71)) )
              
              # Remove small blobs ?? Want to confirm
              res0_thresholded_area_open = area_opening(res0_thresholded_where, area_threshold = 1500, connectivity= 8) # Removes small lines
              
              # Dilate remaining valids
              res0_thresholded_where_dil = dilation(res0_thresholded_area_open, rectangle(7,70)) #10,100 previous (y was too much)
              
              
              
              # (3.) Remove above and below artifacts
              res0_thresholded_where_dil[:20,:] = 0 # Set all prediction before 20th row to zero          
              # Better to truncate after thresholding to accomodate negative prob value models
              res0_thresholded_where_dil[stop_val:,: ] = 0          
               
              cbin1 = custom_binarize(res02,res0_thresholded_where_dil, closeness = 10, return_segment = False) 
              
              cbin_rmv_objects = remove_small_objects(cbin1.astype('bool'),min_size= 15, connectivity=4)
              cbin_rmv_holes = remove_small_holes(cbin_rmv_objects.astype('bool'),connectivity=8)
              cbin_dilate_again = dilation(cbin_rmv_holes, rectangle(2,100))
              cbin = custom_binarize(res0_dil,cbin_dilate_again, closeness = 10, return_segment = False)
              
              
              # thresh = 12
              # vec_layer = create_vec_layer( res0_binarized_thin_rbin,thresh);      
              # vec_layer[vec_layer==0] = np.nan
              
              
              if 1:
                  if not os.path.exists(os.path.join(out_dir,L_folder, model_name,'debug_images2')):
                      os.makedirs(os.path.join(out_dir,L_folder,model_name,'debug_images2'), exist_ok=True)        
            
                  f, axarr = plt.subplots(1,11,figsize=(20,20))
                      
                  axarr[0].imshow(a01.squeeze(),cmap='gray_r')
                  axarr[0].set_title( f'Echo {os.path.basename(curr_data)}') #.set_text
                    
                  axarr[1].imshow(res0,cmap='viridis' ) #, norm=colors.SymLogNorm(linthresh=0.03, vmin=.85*a01.min(), vmax=a01.max())
                  axarr[1].set_title('Model out')
                  
                  axarr[2].imshow(res02,cmap='viridis' ) #, norm=colors.SymLogNorm(linthresh=0.03, vmin=.85*a01.min(), vmax=a01.max())
                  axarr[2].set_title('Regions')                   

                  axarr[3].imshow(res0_thresholded_where.astype(bool).astype(int), cmap='viridis' )
                  axarr[3].set_title('res0_thresholded_where') 
                  
                  axarr[4].imshow(res0_thresholded_area_open.astype(bool).astype(int), cmap='viridis' )
                  axarr[4].set_title('res0_thresholded_area_open') 
                  
                  axarr[5].imshow(res0_thresholded_where_dil.astype(bool).astype(int), cmap='viridis' )
                  axarr[5].set_title('res0_thresholded_where_dil') 
                  
                  axarr[6].imshow(cbin1, cmap='viridis', norm=colors.SymLogNorm(linthresh=0.03, vmin=.85*res0.min(), vmax=res0.max()) )
                  axarr[6].set_title("cbin1") 
                  
                  axarr[7].imshow(cbin_rmv_objects, cmap='viridis', norm=colors.SymLogNorm(linthresh=0.03, vmin=.85*res0.min(), vmax=res0.max()) )
                  axarr[7].set_title("cbin_rmv_objects") 
                  
                  axarr[8].imshow(cbin_rmv_holes, cmap='viridis', norm=colors.SymLogNorm(linthresh=0.03, vmin=.85*res0.min(), vmax=res0.max()) )
                  axarr[8].set_title("cbin_rmv_holes") 
                  
                  axarr[9].imshow(cbin_dilate_again, cmap='viridis', norm=colors.SymLogNorm(linthresh=0.03, vmin=.85*res0.min(), vmax=res0.max()) )
                  axarr[9].set_title("cbin_dilate_again") 
                  
                  axarr[10].imshow(cbin, cmap='viridis', norm=colors.SymLogNorm(linthresh=0.03, vmin=.85*res0.min(), vmax=res0.max()) )
                  axarr[10].set_title("cbin") 
                  
                  # axarr[11].imshow(a01.squeeze(),cmap='gray_r' )
                  # axarr[11].plot(vec_layer.T)
                  # axarr[11].set_title(f"Plotted_layer_output_names{mod_idx}") 
                  
                  save_fig_path = os.path.join(out_dir,L_folder,model_name,'debug_images2',base_name)
                  save_fig_path,_ =  os.path.splitext(save_fig_path)
                  plt.savefig(save_fig_path+'.png')        
                  
                  plt.close()
                    

                    
   
              

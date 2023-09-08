# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 21:30:25 2023

@author: i368o351
"""
import string
import matplotlib as mpl
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

from custom_binarize import custom_binarize
from make_vec_layer import make_vec_layer


import os

from scipy.io import loadmat,savemat
from scipy.ndimage import median_filter as sc_med_filt




model_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data\EchoViT_paper\EchoViT_paper_OKAY_binary_30_December_22_2204.h5'
model = tf.keras.models.load_model(model_path)


output_base_path = r'Y:\ibikunle\Python_Project\Fall_2021\Predictions_Folder\EchoViT_out2'

ab = list( mpl.cycler(mpl.rcParams['axes.prop_cycle']) )
clr = [ item['color'] for item in ab if item['color'] != '#7f7f7f' ] # Remove illegible color
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=clr) 

alphas = string.ascii_lowercase

intpolate = 'antialiased'

base_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data\test_L_files' 
echos = [r'L1\20120330_03_0003_2km.mat', r'L2\20120330_04_0157_5km.mat'] #, r'L3\20120330_04_1170_2km.mat'

output_names = ['FastEmbed','SlowEmbed','Cropped']

# f, axarr = plt.subplots(3,4,figsize=(25,25))

# f2, axarr2 = plt.subplots(3,5,figsize=(25,25))

for iter,echo in enumerate(echos):
    fp = os.path.join(base_path,echo)
    predict_data = loadmat(fp)
    file = os.path.splitext( os.path.basename(echo) )[0]
    
    a01,a_gt0 = predict_data['echo_tmp'], predict_data['raster']
    res0_all = model.predict ( np.expand_dims(np.expand_dims(a01,axis=0),axis=3) ) 
    Nt,Nx = a01.shape   
    
    # axarr[iter,0].imshow(a01.squeeze(),cmap='gray_r', interpolation=intpolate)
    # axarr[iter,0].set_title('Echogram')
    plt.figure(figsize=(20,25))
    plt.imshow(a01.squeeze(),cmap='gray_r', interpolation=intpolate) 
    outpath_one = os.path.join(output_base_path,'echo_'+ file+'_echo.png')
    plt.savefig(outpath_one,transparent=True, bbox_inches='tight') 
    plt.close()
    
    # axarr2[iter,0].imshow(a01.squeeze(),cmap='gray_r', interpolation=intpolate)
    # axarr2[iter,0].set_title('Echogram')
    
    # axarr2[iter,1].imshow(a01.squeeze(),cmap='gray_r', interpolation=intpolate)  
    # axarr2[iter,1].plot(predict_data['vec_layer'].T)
    # axarr2[iter,1].set_title('Overlaid GT')
    
    plt.figure(figsize=(20,25))
    _ = plt.imshow(a01.squeeze(),cmap='gray_r', interpolation=intpolate)
    _ = plt.plot(predict_data['vec_layer'].T)
    _ = plt.title('Overlaid GT')
    _= plt.tick_params( which='both', bottom=False, top=False, left=False, labelbottom=False,labelleft=False)
    outpath_one = os.path.join(output_base_path,'echo_overlaid_GT'+ file+'_echo.png')
    plt.savefig(outpath_one,transparent=True, bbox_inches='tight') 
    plt.close()
    
    
    for mod_idx in range(1,4):
        res0 = res0_all[mod_idx-1]
        model_name = output_names[mod_idx-1]
        
        res0 = res0.squeeze() 
        
        plt.figure(figsize=(20,25))
        _ = plt.imshow(res0,cmap='viridis')
        _= plt.tick_params( which='both', bottom=False, top=False, left=False, labelbottom=False,labelleft=False)
        outpath_one = os.path.join(output_base_path,'actv_map_'+ file+output_names[mod_idx-1]+'_.png')
        plt.savefig(outpath_one,transparent=True, bbox_inches='tight') 
        plt.close()    
        
        #axarr[iter,mod_idx].imshow(res0,cmap='viridis', norm=colors.SymLogNorm(linthresh=0.03, vmin=.85*a01.min(), vmax=a01.max()), interpolation=intpolate)
        
        binarize_threshold = np.percentile(res0,75)
            
        res0_final = np.where(res0>binarize_threshold,1,0) 
        cbin = custom_binarize(res0,res0_final, closeness = 8, return_segment = False)
        
        
        res0_final1 = np.arange(1,Nt+1).reshape(Nt,1) *  cbin  
        
      
        # How correct is create_vec_layer??
        thresh = {'constant': 40}
        vec_layer = make_vec_layer(res0_final1,thresh);      
        vec_layer[vec_layer==0] = np.nan
      
        
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
        short_layers = np.argwhere( np.sum(np.isnan(new_layer_filtered),axis = 1) > round(.5*Nx) )
        short_layers = [elem for a in short_layers for elem in a]
        
        other_layers = list( set(range(new_layer_filtered.shape[0])) - set(short_layers) )
        
        if other_layers:
            for rw_idx in other_layers:
                if np.any( np.diff(new_layer_filtered[rw_idx]) > 10):
                    short_layers.append(rw_idx)
        
        new_layer_filtered = np.delete(new_layer_filtered,short_layers,axis = 0)         
        
        # axarr2[iter,mod_idx+1].imshow(a01.squeeze(),cmap='gray_r', interpolation=intpolate) 
        # axarr2[iter,mod_idx+1].plot(new_layer_filtered.T)
        
        plt.figure(figsize=(20,25))
        _ = plt.imshow(a01.squeeze(),cmap='gray_r', interpolation=intpolate)
        _ = plt.plot(new_layer_filtered.T)
        _= plt.tick_params( which='both', bottom=False, top=False, left=False, labelbottom=False,labelleft=False)
        outpath_one = os.path.join(output_base_path,'tracked_'+ file+output_names[mod_idx-1]+'_.png')
        plt.savefig(outpath_one,transparent=True, bbox_inches='tight') 
        plt.close()
        
        
#         if mod_idx > 0:
#             axarr2[iter,mod_idx].set_xticks([])
#             axarr2[iter,mod_idx].set_yticks([]) 
#             axarr2[iter,mod_idx].set_xlabel( "(" + alphas[mod_idx+1] + ")", size = 15)
            
#             axarr[iter,mod_idx].set_xticks([])
#             axarr[iter,mod_idx].set_yticks([]) 
#             axarr[iter,mod_idx].set_xlabel( "(" + alphas[mod_idx] + ")", size = 15)
            
# plt.subplots_adjust(bottom=0.15, wspace=0.05)
# plt.show()
        
    

    


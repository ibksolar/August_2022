# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 21:32:08 2023

@author: i368o351
"""


import os
from scipy.io import loadmat,savemat
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
# import matplotlib.pylab as pylab
import string

import glob



#==============================================================================
# Paths
#==============================================================================
echo_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data\test_data'  # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'
model_out_path = r'Y:\ibikunle\Python_Project\Fall_2021\Predictions_Folder\DL_models_predictions_folder_final'
output_base_path = r'Y:\ibikunle\Python_Project\Fall_2021\Predictions_Folder\DL_models_predictions_folder_final\Paper_final_figure2'


files_to_plot = ['0001_2km', '0001_5km', '0001_10km', '0021_5km']
files_to_plot = ['20120330_03_'+ elem for elem in files_to_plot ]

alphas = string.ascii_lowercase

models = ['SimpleUNet', 'AttUNet','DeepLab', 'FCN', 'ensemble',]

if 0:    
    for file in files_to_plot:
        
        echo_file_fp = os.path.join(echo_path,file)
        echo_file = loadmat(echo_file_fp)
        
        ## Create Echogram plot 
        
        _ = plt.imshow(echo_file['echo_tmp'], cmap = 'gray_r')
        _ = plt.title(f'Echo {file}')
        outpath = os.path.join(output_base_path, file+'_echo.png')
        plt.savefig(outpath, transparent=True, bbox_inches='tight')
        plt.close()
        
        for model in ['AttUNet','DeepLab','ensemble','FCN','SimpleUNet']:
            pred_file_fp = os.path.join(output_base_path,'L1',model, file+'.mat')
            pred_file = loadmat(pred_file_fp)
        
        
            ## Create plots 
            
            # GT ( Only plot the first time)
            if model == 'AttUNet':
                
                _ = plt.imshow(echo_file['echo_tmp'], cmap = 'gray_r')
                _ = plt.plot(pred_file['GT_layer'].T)
                _ = plt.title('Ground truth')
                outpath = os.path.join(output_base_path, model +'_GT_'+ file+'_echo.png')
                plt.savefig(outpath,transparent=True, bbox_inches='tight')
                plt.close()
            
            
            # Binary output
            _ = plt.imshow(pred_file['binary_output'].astype(bool).astype(int), cmap='hot')
            _ = plt.title('Binary output')
            outpath = os.path.join(output_base_path, model +'_binary_'+ file+'_echo.png')
            plt.savefig(outpath,transparent=True, bbox_inches='tight' ) #transparent=True
            plt.close()
            
            # Segmented
            _ = plt.imshow(echo_file['echo_tmp'], cmap = 'gray_r')
            _ = plt.plot(pred_file['vec_layer'].T)
            _ = plt.title('Segmentation layers')
            outpath = os.path.join(output_base_path, model +'_segment_'+ file+'_echo.png')
            plt.savefig(outpath,transparent=True, bbox_inches='tight')
            plt.close()
        

if 1:            
    for file in files_to_plot:
        
        echo_file_fp = os.path.join(echo_path,file)
        echo_file = loadmat(echo_file_fp)        

        bins = {}
        segmented = {}
        for model in models:
            pred_file_fp = os.path.join(model_out_path,'L1',model, file+'.mat')
            pred_file = loadmat(pred_file_fp)
            
            bins[model] = pred_file['binary_output'].astype(bool).astype(int) 
            segmented[model] = pred_file['filtered_vec_layer']
            
        
        ## Create plots 
        # Binary output
        
        plot_data = [ echo_file['echo_tmp'],echo_file['raster'].astype(bool).astype(int),
                      bins[models[0]], bins[models[1]],bins[models[2]],bins[models[3]], bins[models[4]]  ]
        
        cmaps = ['gray_r'] + ['gray']*6
        
        f, axarr = plt.subplots(1,7,figsize=(20,20))
        
        for plt_iter in range(len(cmaps)):
            axarr[plt_iter].imshow(plot_data[plt_iter], cmap = cmaps[plt_iter])            
            if plt_iter > 0:
                axarr[plt_iter].set_xticks([])
                axarr[plt_iter].set_yticks([]) 
                axarr[plt_iter].set_xlabel( "(" + alphas[plt_iter] + ")", size = 15)
            # else:
            #     axarr[plt_iter].set_title(f'Echo {file}', size = 15) 
                

        outpath = os.path.join(output_base_path,'binary_'+ file+'_echo.png')
        plt.savefig(outpath,transparent=True, bbox_inches='tight')
        plt.close()
        
        
        # Segment output
        f, axarr = plt.subplots(1,7,figsize=(20,20))
        # Set the default color cycle
        ab = list( mpl.cycler(mpl.rcParams['axes.prop_cycle']) )
        clr = [ item['color'] for item in ab if item['color'] != '#7f7f7f' ] # Remove illegible color
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=clr) 
        
        for plt_iter in range(len(cmaps)):
            axarr[plt_iter].imshow(plot_data[0], cmap = 'gray_r') 
            
            # if plt_iter == 0:                
            #     axarr[plt_iter].set_title(f'Echo {file}')

            if plt_iter == 1:
                axarr[plt_iter].plot(echo_file['vec_layer'].T)                
                axarr[plt_iter].set_xticks([])
                axarr[plt_iter].set_yticks([]) 
                axarr[plt_iter].set_xlabel( "(" + alphas[plt_iter] + ")", size = 12)
            elif plt_iter > 1:
                axarr[plt_iter].plot(segmented[models[plt_iter-2]].T)
                axarr[plt_iter].set_xticks([])
                axarr[plt_iter].set_yticks([]) 
                axarr[plt_iter].set_xlabel( "(" + alphas[plt_iter] + ")", size = 12)

        outpath = os.path.join(output_base_path,'segment_'+ file+'_echo.png')
        plt.savefig(outpath,transparent=True, bbox_inches='tight')
        plt.close()
        

        
## Plot for Single echogram to show data, bin_map and seg_map
plt.clf()
f, axarr = plt.subplots(1,3, figsize=(25,25), gridspec_kw = {'wspace':0, 'hspace':0} )

# plt.figure(figsize=(25,25))
# gs1 = mpl.gridspec.GridSpec(1, 3)
# gs1.update(wspace=0.005, hspace=0.005)

axarr[0].imshow(echo_file['echo_tmp'],cmap='gray_r', aspect='auto')
axarr[0].set_title('Echogram data',size = 20) #.set_text
axarr[0].tick_params(labelsize=8)
axarr[0].set_aspect('equal')

axarr[1].imshow(echo_file['raster'].astype(bool).astype(int), cmap='gray', aspect='auto')
axarr[1].set_title('Binary map',size = 20)
# axarr[1].set_aspect('equal')
axarr[1].set_xticks([])
axarr[1].set_yticks([]) 

f3 = axarr[2].imshow(echo_file['semantic_seg'], cmap='bone_r', aspect='auto' )
axarr[2].set_title('Segmentation map',size = 20) 
# axarr[2].set_aspect('equal')
axarr[2].set_xticks([])
axarr[2].set_yticks([]) 
plt.colorbar(f3,ax = axarr[2], fraction= 0.1 )
outpath_one = os.path.join(output_base_path,'segmentation_map_'+ file+'_echo.png')

f.subplots_adjust(wspace=0.05, hspace=0)

plt.savefig(outpath_one,transparent=True, bbox_inches='tight')    



#### Yet another
plt.clf()
N_plot = np.arange(3) + 231
f, axarr = plt.subplots(1,len(N_plot), figsize=(50,50), gridspec_kw = {'wspace':0, 'hspace':0} )

plot_data = [echo_file['echo_tmp'],echo_file['raster'].astype(bool).astype(int),echo_file['semantic_seg']]
cmap_list = ['gray_r', 'gray','bone_r']
for idx,plt_ind in enumerate(N_plot):
    _ = plt.subplot(plt_ind)
    _ = plt.imshow(plot_data[idx], cmap = cmap_list[idx], aspect='auto')
    # plt.title()
f.subplots_adjust(hspace=0.0, wspace=.05)
plt.show()    





base_L_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data\test_L_files'

all_L1 = glob.glob( os.path.join(base_L_path,'L1','*.mat') )
bn_L1 = [ os.path.basename(item) for item in all_L1 ]

all_L2 = glob.glob( os.path.join(base_L_path,'L2','*.mat') )
bn_L2 = [ os.path.basename(item) for item in all_L2 ]

all_L3 = glob.glob( os.path.join(base_L_path,'L3','*.mat') )
bn_L3 = [ os.path.basename(item) for item in all_L3 ]

## Bar plot 
dst = ['2km','5km','10km','20km','50km']
all_train_dst = [ 6577, 2776, 1299, 439, 211 ]
all_test_dst = [ 911, 366, 15]

L1_count = [81, 31, 15]
L2_count = [752,297 ]
L3_count = [78,38]































  
        

        
        
        
       
    


































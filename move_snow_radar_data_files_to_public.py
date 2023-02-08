# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 12:01:49 2022

@author: i368o351
"""

import os
import glob
# from scipy.io import loadmat,savemat
import mat73


######################################################################
# Paths
######################################################################
base_path = r'X:\ct_data\snow\2012_Greenland_P3' 
out_dir = r'X:\public\data\temp\internal_layers\NASA_OIB_test_files\image_files\snow\SR_Dataset_v1'

lean_data_train_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data\train_data'
lean_data_test_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data\val_data' # Val and test intentionally swapped
lean_data_val_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data\test_data'


base_echo_path = os.listdir(base_path)
base_echo_path = sorted( [ elem for elem in base_echo_path if 'attention_dataset' in elem ] )

# all_base_echo_path = [ glob.glob(os.path.join(base_path,idx,'*.mat')) for idx in base_echo_path]


all_orig_mat_files = []
for root, dirs, files in os.walk(base_path, topdown=True):
    for file in files:
        if 'attention_dataset' in root:
            if '.mat' in file:
                all_orig_mat_files.append(os.path.join(root,file))


lean_all_train = glob.glob(os.path.join(lean_data_train_path,'*.mat'))
lean_all_test = glob.glob(os.path.join(lean_data_test_path,'*.mat'))
lean_all_val = glob.glob(os.path.join(lean_data_val_path,'*.mat'))


# ===========================================================================
# Copy_and_move_DS_files function
# ===========================================================================

def simple_copy(source,destination):
    import shutil
    # Copy the content of
    # source to destination
     
    try:
        shutil.copy(source, destination)
        #print("File copied successfully.")
     
    # If source and destination are same
    except shutil.SameFileError:
        print("Source and destination represents the same file.")
     
    # If there is any permission issue
    except PermissionError:
        print("Permission denied.")
     
    # For other errors
    except:
        print("Error occurred while copying file.")



def Copy_and_move_DS_files(lean_files, all_files, output_base_dir = out_dir, train_test_flag='train_data', correct_file_size = False):
    """    

    Parameters
    ----------
    lean_files : TYPE: list
        DESCRIPTION.
        List of full path of either train, test or validation in reduced file size and content (lean) path
        
    all_files : TYPE: list
        DESCRIPTION.
        List of all .mat file with large size and full content (lean) path    
            
    output_base_dir : TYPE: string
        DESCRIPTION.
        base directory of the output before "train", "test" or "val"
    
    train_test_flag : TYPE, optional when creating train files
        DESCRIPTION. The default is 'train_data'. Typically one of 'train_data','test_data' or 'val_data'
        
    correct_file_size : TYPE, boolean: True or false
        DESCRIPTION. Choose whether to alter Data size to create fixed sized outputs or not

    Returns
    -------
    None.

    """
    for idx,curr_lean_file in enumerate(lean_files):
        curr_base_name,_ = os.path.splitext( os.path.basename(curr_lean_file) )
        
        dist_idx = curr_base_name.rfind('_') # Distance index
        
        fname,dist = curr_base_name[:dist_idx], curr_base_name[dist_idx+1:]
        
        mat_file_path =  [item for item in all_files if fname in item and dist in item]
        
        # Print status
        print(f'{idx} of {len(lean_files)} - {curr_base_name} ')
        
        if not mat_file_path:
            print(f'Could not find {curr_base_name}')
        else:
            mat_file_path = mat_file_path[0]
            base_path,_ = os.path.split(mat_file_path)
            
            png_path = os.path.join(base_path,'img_'+ fname +'.png')
            layer_bin_path = os.path.join(base_path,'layer_bin_' + fname + '.png')
            layer_seg_path = os.path.join(base_path,'layer_seg_'+ fname +'.png')
            
            if correct_file_size:
                l_mat_file = mat73.loadmat(mat_file_path) #Loaded mat file            
                Data = l_mat_file['Data']
                
                curr_Nt,curr_Nx = Data.shape            
                curr_Nt = curr_Nt if curr_Nt % 32 == 0 else ((curr_Nt//32) )*32     
                curr_Nx = curr_Nx if curr_Nx % 32 == 0 else ((curr_Nx//32))*32 -1
                
                l_mat_file['Data'] = Data[:curr_Nt, :curr_Nx]
                
                # TO DO: Complete Data size adjustment later
            
            ## Check and create out_paths
            
            base_out_path = os.path.join(output_base_dir,train_test_flag)
            if not os.path.exists(base_out_path):
                os.mkdir(base_out_path) 
            
            # Copy .mat    
            mat_out_path = os.path.join(base_out_path,curr_base_name+'.mat') 
            simple_copy(mat_file_path,mat_out_path)
            
            dist2 = "_" + dist
            
            # Copy image .png
            png_out_path = os.path.join(base_out_path,'img_'+ fname + dist2 + '.png')
            simple_copy(png_path,png_out_path)
    
            # Copy bin .png 
            layer_bin_out_path = os.path.join(base_out_path,'layer_bin_' + fname + dist2 +'.png')
            simple_copy(layer_bin_path,layer_bin_out_path)
            
            # Copy seg .png 
            layer_seg_out_path = os.path.join(base_out_path,'layer_seg_'+ fname + dist2 +'.png')
            simple_copy(layer_seg_path,layer_seg_out_path)

   
# Copy/Create training_set files
Copy_and_move_DS_files(lean_all_train,all_orig_mat_files)
print('Finished transferring training set')


# Copy/Create testing_set files
Copy_and_move_DS_files(lean_all_test,all_orig_mat_files,train_test_flag='test_data')
print('Finished transferring testing set')


# Copy/Create validation_set files
Copy_and_move_DS_files(lean_all_val,all_orig_mat_files,train_test_flag='val_data')
print('Finished transferring validation set')




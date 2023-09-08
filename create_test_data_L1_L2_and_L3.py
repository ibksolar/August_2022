# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 09:41:13 2023
Script to create oiginal test_data L1,L2 and L3

@author: Ibikunle
"""

import os
import glob

test_data_base_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data\test_data'
destination_base_path =  r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data\test_L_files'


L1_base = r'Y:\ibikunle\Python_Project\Fall_2021\Predictions_Folder\DL_models_predictions_folder_final\L1\AttUNet\*.mat'
L2_base = r'Y:\ibikunle\Python_Project\Fall_2021\Predictions_Folder\DL_models_predictions_folder_final\L2\AttUNet\*.mat'
L3_base = r'Y:\ibikunle\Python_Project\Fall_2021\Predictions_Folder\DL_models_predictions_folder_final\L3\AttUNet\*.mat'


L1_file_names = glob.glob(L1_base)
L1_file_names = [os.path.basename(elem) for elem in L1_file_names]

L2_file_names = glob.glob(L2_base)
L2_file_names = [os.path.basename(elem) for elem in L2_file_names]

L3_file_names = glob.glob(L3_base)
L3_file_names = [os.path.basename(elem) for elem in L3_file_names]


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
        
        

def move_files(source_base_path, destination_base_path, level ="L1"):
    """    
    Parameters
    ----------
    source_base_path : TYPE: String
        DESCRIPTION: filepath of source
        
    level : TYPE: String, Default -"L1"
        DESCRIPTION. L1,L2 or L3
        
    destination_base_path : TYPE: String
        DESCRIPTION: base file path of the destination folder

    Returns
    -------
    None.

    """
    if level == "L1":
        Level_file_names = L1_file_names  
    else:
        Level_file_names = L2_file_names if level =="L2" else L3_file_names
    
    destination_path = os.path.join(destination_base_path,level)
    
    for file_name in Level_file_names:
        source_file = os.path.join(source_base_path,file_name)
        try:
            simple_copy(source_file, destination_path)
            print(f' Copied {file_name} to {level} destination path')
        except:
            print(f' Something went wrong! Could not copy {file_name} to {level} destination path')
            


# Move files to levels using functions
Level_list =["L1","L2","L3"]

for level in Level_list:
    move_files(test_data_base_path,destination_base_path,level=level)
    print(f'Finished moving {level} files')
        
    




































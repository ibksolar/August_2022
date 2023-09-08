# -*- coding: utf-8 -*-
"""
Created 3rd June,2022

@author: i368o351
"""

from tensorflow.keras import layers
from tensorflow import keras
#from keras import backend as K

import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm,colors

import os
import random
from scipy.io import loadmat
from scipy.ndimage import median_filter as sc_med_filt

# from keras.metrics import MeanIoU
# from sklearn.metrics import roc_auc_score

import glob
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,ReduceLROnPlateau
from datetime import datetime

# from focal_loss import SparseCategoricalFocalLoss

import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()

## GPU Config
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      #tf_config = tf.ConfigProto(allow_soft_placement=False)
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)  
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

model_name = 'echo_vit_3D_raster'
use_wandb = True
time_stamp = '15_April_23_1233' #datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')

if use_wandb:    
    ## WandB config
    import wandb
    from wandb.keras import WandbCallback  
    wandb.init( project="my-test-project", entity="ibksolar", name = model_name+time_stamp,config ={})
    config = wandb.config
else:
    config ={}




# PATHS
# Path to data
base_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data'  # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'
train_aug = os.path.join(base_path,'augmented_plus_train_data\*.mat')
train_path = os.path.join(base_path,'train_data\*.mat')
val_path = os.path.join(base_path,'val_data\*.mat')
test_path = os.path.join(base_path,'test_data\*.mat')   

# Create tf.data.Dataset
config['Run_Note'] = 'Training Regression network for layers_first_try'
config['batch_size'] = 4

config['img_y'] = 1664 #1664 , 416
config['img_x'] = 256 #256, 64
config['dropout_rate'] = 0.2

# Training params
#config={}
config['img_channels'] = 1

config['num_classes'] = 1
config['epochs'] = 150
config['learning_rate'] = 1e-3
config['base_path'] = base_path
SEED = 42
AUTO = tf.data.experimental.AUTOTUNE



# =============================================================================
## Function for creating dataloader

def read_mat(filepath):
    def _read_mat(filepath):
        
        dtype = tf.float16
        
        filepath = bytes.decode(filepath.numpy())      
        mat_file = loadmat(filepath)
        echo = tf.cast(mat_file['echo_tmp'], dtype= dtype) #, dtype=tf.float64
        
        echo = tf.expand_dims(echo, axis=-1)
        
        layer = tf.cast( tf.cast(mat_file['raster'], dtype=tf.bool), dtype = dtype)
        layer = tf.expand_dims(layer, axis=-1)
        
        layer3D = tf.cast( tf.cast(mat_file['raster_3D'], dtype=tf.bool), dtype = dtype)        
            
        shape0 = echo.shape        
        return echo,layer,layer3D,np.asarray(shape0)     
    
    output = tf.py_function(_read_mat,[filepath],[tf.half,tf.half,tf.half, tf.int32]) #,tf.double
    shape = output[3]
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([config['img_y'],config['img_x'], config['img_channels'] ])
    
    data1 = output[1]  
    data1.set_shape([config['img_y'],config['img_x'], config['img_channels'] ])
    
    data2 = output[2] 
    data2.set_shape([config['img_y'],config['img_x'],3]) #,30, ,config['num_classes']    
    
    return data0,{'raster':data1, 'raster3D':data2}

train_ds = tf.data.Dataset.list_files(train_aug,shuffle=True) #'*.mat'
train_ds = train_ds.map(read_mat,num_parallel_calls=8)
train_ds = train_ds.batch(config['batch_size'],drop_remainder=True).prefetch(AUTO) #.shuffle(buffer_size = 100 * config['batch_size'])

# No augmentation for testing and validation
val_ds = tf.data.Dataset.list_files(val_path,shuffle=True)
val_ds = val_ds.map(read_mat,num_parallel_calls=8)
val_ds = val_ds.batch(config['batch_size'],drop_remainder=True).cache().prefetch(AUTO)

test_ds = tf.data.Dataset.list_files(test_path,shuffle=True)
test_ds = test_ds.map(read_mat,num_parallel_calls=8)
test_ds = test_ds.batch(config['batch_size'],drop_remainder=True).cache().prefetch(AUTO)


## Confirm data loader is working well
# train_shape = [ ( tf.shape(item[0]).numpy(),tf.shape(item[1]).numpy() ) for item in train_ds.take(1) ]
# train_shape = train_shape[0]
# print(f' X_train train shape {train_shape[0]}')
# print(f' Training target shape {train_shape[1]}')


config['embed_dim'] = embed_dim =  config['img_y'] # train_shape[0][1] #416
config['num_patches'] = num_patches = config['img_x'] #train_shape[0][2] #64 


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation= tf.nn.gelu) (x)
        x = layers.Dropout(dropout_rate)(x)
    return x


# Model hyper-param

num_heads = 12
dense_dim = 512

transformer_units = [
    embed_dim * 4,
    embed_dim,
]

transformer_layers = 20

proj_head_units = [2048, 1024, 512, 64] #2048,
mlp_head_units = [ 512,256] #mlp_head_units = [ 2048,1024, 512, 64]  # Size of the dense layers , 2048, 1024,

kernel_init = 'glorot_normal'

input_shape = (config['img_y'], config['img_x'], config['img_channels'])
# input_shape = (None, None, config['img_channels'])

strategy = tf.distribute.MirroredStrategy( cross_device_ops=tf.distribute.HierarchicalCopyAllReduce() )
#print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    
    used_dtype = tf.float32
    
    def transformer_unit(encoded_patches, transformer_layers = transformer_layers):    
        for _ in range(transformer_layers):
            
            encoded_patches = tf.cast(encoded_patches, dtype= used_dtype)
            
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim, dropout=config['dropout_rate']
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            
            # Skip connection 2.
            x4 = layers.Add()([x3, x2])
            
            # Feed forward
            mlp_out = mlp(x4,mlp_head_units,dropout_rate=0.2)
            x5 = layers.Add()([mlp_out, x4])
            
            encoded_patches = layers.LayerNormalization(epsilon=1e-6)(x5)
            
            encoded_patches = tf.cast(encoded_patches, dtype = used_dtype)
            
            return encoded_patches
    
    
    
    inputs = layers.Input(shape = input_shape, dtype = used_dtype)    
    # ====================================================================
    # Create each Embed and ResNet 50    
    # ====================================================================
    
    ## ResUNet input
    ResNet_50_model = sm.Unet(backbone_name='resnet50', encoder_weights='imagenet', encoder_freeze=True)
    in1 = layers.Conv2D(3, (3, 3), activation='relu', kernel_initializer= kernel_init, padding='same', dtype =used_dtype )(inputs)
    in1 = ResNet_50_model(in1)
    in1 = tf.cast(in1, dtype=used_dtype)
    in2 = layers.Conv2D(1, (config['img_y'],1), activation='relu', kernel_initializer= kernel_init, padding='same' , dtype =used_dtype)(in1)
    
    
    ## Row Embed    
    rw_embed = tf.image.extract_patches(images = in2, sizes=[1,1,config['img_x'],1], strides=[1,1,config['img_x'],1], rates=[1,1,1,1], padding ="SAME")
    rw_embed = tf.cast( layers.Reshape(input_shape[:-1])(rw_embed) , dtype =used_dtype)
    
    ## Column Embed
    col_embed = layers.Conv2D(1, (config['img_y'],1), activation='relu', kernel_initializer= kernel_init, padding='same' , dtype =used_dtype)(in2)
    
    ## Patch Embed
    # patch_embed = tf.image.extract_patches(images = in2, sizes=[1,config['img_y']//4,config['img_x']//4,1], strides=[1,config['img_y']//4,config['img_x']//4,1], rates=[1,1,1,1], padding ="SAME")
    # patch_embed = tf.cast( layers.Reshape(input_shape[:-1])(patch_embed) , dtype =used_dtype)   
    
    
       
    # ====================================================================
    # Add positional encoding    
    # ====================================================================    
    
    pos_embed = layers.Embedding(input_dim = input_shape[0], output_dim = input_shape[1], input_length = input_shape[0] ) (tf.range(start=0,limit=input_shape[0],delta=1))
    pos_embed = tf.cast(pos_embed, dtype =used_dtype)
    
    # (a.) Row embed + pos_encode
    rw_embed = tf.cast(rw_embed + pos_embed , dtype=used_dtype)
    
    # (b.) Col embed + pos_encode
    col_embed = tf.reduce_mean(col_embed,axis = -1)
    col_embed = tf.cast( col_embed + pos_embed , dtype=used_dtype)
    
    # (c.) Patch embed + pos_encode
    # patch_embed = tf.cast( patch_embed + pos_embed , dtype=used_dtype)
    
    # (d.) ResUNet + pos_encode
    x = tf.reduce_mean(in2,axis = -1)
    Resnet_patch = tf.cast( x + pos_embed , dtype=used_dtype)
    
    
   
    # ====================================================================
    # Pass as input to Transformer   
    # ====================================================================  
    rw_transform = transformer_unit(rw_embed)
    rw_transform = layers.LayerNormalization(epsilon=1e-6)(rw_transform)
    
    col_transform = transformer_unit(col_embed)
    col_transform = layers.LayerNormalization(epsilon=1e-6)(col_transform)

    # patch_transform = transformer_unit(patch_embed)
    # patch_transform = layers.LayerNormalization(epsilon=1e-6)(patch_transform)

    ResNet_transform = transformer_unit(Resnet_patch)
    ResNet_transform = layers.LayerNormalization(epsilon=1e-6)(ResNet_transform)
    
    combined = rw_transform + col_transform + ResNet_transform   

    
    used_activation = 'sigmoid' if config['num_classes'] == 1 else 'softmax'
    
    # ====================================================================
    # Raster 3D Output convolutions ( to be used for regression)   
    # ====================================================================  
    # Create output convolution 
    output_convolution = tf.keras.Sequential()
    output_convolution.add(layers.Conv2D(config['num_classes']*5, (17,15), activation='relu', padding="same", dtype = used_dtype) )    
    output_convolution.add( layers.Conv2D(config['num_classes']*5, 3, activation="relu", padding="same", dtype = used_dtype ) )
    output_convolution.add( layers.Dropout(config['dropout_rate']) )
    output_convolution.add( layers.Conv2D( 3, (1,1), padding="same", dtype = used_dtype, name= "raster3D") ) #, activation= used_activation
    
    
    # ====================================================================
    # Expand dims and outputs 
    # ==================================================================== 
    # rw_transform = tf.expand_dims(rw_transform, axis=-1)
    # rw_embed_out = output_convolution(rw_transform)
    
    # col_transform = tf.expand_dims(col_transform, axis=-1)
    # col_embed_out = output_convolution(col_transform)
    
    # # patch_transform = tf.expand_dims(patch_transform, axis=-1)
    # # patch_embed_out = output_convolution(patch_transform)    
    
    # ResNet_transform = tf.expand_dims(ResNet_transform, axis=-1)
    # ResNet_embed_out = output_convolution(ResNet_transform)
    
    combined = tf.expand_dims(combined, axis=-1)
    combined = layers.Conv2D(config['num_classes']*5, (17,15), activation='relu', padding="same", dtype = used_dtype) (combined)
    combined = layers.Conv2D(config['num_classes'], (17,15), activation='relu', padding="same", dtype = used_dtype) (combined)
    combined_out = layers.Conv2D( 3, (1,1), padding="same", dtype = used_dtype, name= "raster3D")(combined)
     
    
    
    # ====================================================================
    # Create binary segmentation output using combined before regression output
    # Single binary output
    # ====================================================================     
    binary_out = layers.Conv2D(config['num_classes']*5, (17,15), activation='relu', padding="same", dtype = used_dtype) (combined)
    binary_out = layers.Conv2D(config['num_classes'], (17,15), activation='relu', padding="same", dtype = used_dtype) (binary_out)
    binary_out = layers.Conv2D( 1, (1,1), padding="same", dtype = used_dtype, activation= used_activation, name= "raster")(binary_out)  #, activation= used_activation
    
    outputs = [binary_out, combined_out ] #rw_embed_out, col_embed_out, ResNet_embed_out, 
   
    # Return Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    #model =  create_vit_object_detector(input_shape,num_patches,embed_dim,num_heads,transformer_units,transformer_layers,mlp_head_units,)
    
    opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate']) #, clipnorm=0.5
    opt2 = tfa.optimizers.AdamW(weight_decay = 0.001, learning_rate=config['learning_rate'])
    
    ## Print Start time
    config['start_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
    print(f' Start time {config["start_time"]}')
    
    
    logz= f"{config['base_path']}/{model_name}/{config['start_time']}_logs/"
    callbacks = [
       ModelCheckpoint(f"{config['base_path']}//{model_name}//{model_name}_Checkpoint{time_stamp}.h5", save_best_only=True, monitor="val_loss"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.25, patience=10, min_lr=0.00001, verbose= 1),
        EarlyStopping(monitor="val_loss", patience=30, verbose=1), 
        TensorBoard(log_dir = logz,histogram_freq = 1,profile_batch = '1,70', embeddings_freq=50),
        WandbCallback()
    ]
    
    # loss = tf.keras.losses.BinaryCrossentropy() if config['img_channels']==1 else tf.keras.losses.CategoricalCrossentropy()
    loss = keras.losses.MeanSquaredError()
    
    model.compile(optimizer=opt2,#opt
              loss = ["binary_crossentropy", "mean_squared_error"], #, "mean_squared_error", "mean_squared_error", "mean_squared_error"  
             metrics=[ ['accuracy'] ,['mean_squared_error']  ] ) #metrics=['accuracy']                                           
    
    history = model.fit(train_ds, epochs = config['epochs'],validation_data = val_ds, callbacks = callbacks)

## Print End time
config['start_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
print(f' Completion time {config["start_time"]}')

loaded_model = tf.keras.models.load_model(f"{config['base_path']}//{model_name}//{model_name}_Checkpoint{time_stamp}.h5")
                                       

# Save model with proper name
_,acc = model.evaluate(test_ds)
model.save(f"{config['base_path']}//{model_name}//{model_name}_{acc:.2f}_{time_stamp}.h5")

# Custom colormap
custom_cm = cm.BuPu(np.linspace(0,1,30))
custom_cm = colors.ListedColormap(custom_cm[10:,:-1])

## Visualize result of model prediction for "unseen" echogram during training
model_val_data_path = os.path.join(base_path,'test_data\*.mat')
model_val_data = glob.glob(model_val_data_path)

batch_idx = random.randint(1,len(model_val_data)) # Pick any of the default batch

output_names = ['rw_embed_out', 'col_embed_out', 'patch_embed_out', 'ResNet_embed_out', 'combined_out']

from custom_binarize import custom_binarize
from make_vec_layer import make_vec_layer


save_pred = 1

if save_pred:
    L1_files = glob.glob(os.path.join(base_path,r'test_L_files\L1\*.mat') )
    L2_files = glob.glob(os.path.join(base_path,r'test_L_files\L2\*.mat') )
    L3_files = glob.glob(os.path.join(base_path,r'test_L_files\L3\*.mat') )
    test_data = L1_files + L2_files + L3_files
    
    #out_dir = os.path.join(base_path,'predictions_folder')
    out_dir = r'Y:\ibikunle\Python_Project\Fall_2021\Predictions_Folder\EchoViT_out'

L_files = [L1_files,L2_files,L3_files]
L_folders = ['L1','L2','L3']


for idx in range(5):
  predict_data = loadmat(model_val_data[batch_idx+idx])
  a01,a_gt0 = predict_data['echo_tmp'], predict_data['raster']
  
  Nt,Nx = a01.shape
  
  if model.input_shape[-1] > 1:
    a0 = np.stack((a01,)*3,axis=-1)
    res0_all = model.predict ( np.expand_dims(a0,axis=0))
  else:
    res0_all = model.predict ( np.expand_dims(np.expand_dims(a01,axis=0),axis=3) )   
 
  for mod_idx in range(len(res0_all)):
      res0 = res0_all[mod_idx]
      model_name = output_names[mod_idx]
      
      res0 = res0.squeeze()
      
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
      short_layers = np.argwhere( np.sum(np.isnan(new_layer_filtered),axis = 1) > Nx//1.3)
      new_layer_filtered = np.delete(new_layer_filtered,short_layers,axis = 0) 

      pred0_final = sc_med_filt( sc_med_filt(res0_final,size=7).T, size=7, mode='nearest').T
    
    
      f, axarr = plt.subplots(1,7,figsize=(20,20))
          
      axarr[0].imshow(a01.squeeze(),cmap='gray_r')
      axarr[0].set_title( f'Echo {os.path.basename(model_val_data[batch_idx+idx])}') #.set_text
        
      axarr[1].imshow(a01,cmap='viridis', norm=colors.SymLogNorm(linthresh=0.03, vmin=.85*a01.min(), vmax=a01.max()) )
      axarr[1].set_title('Orig echo map')
        
      axarr[2].imshow(res0, cmap='viridis', norm=colors.SymLogNorm(linthresh=0.03, vmin=.85*res0.min(), vmax=res0.max()) )
      axarr[2].set_title(f'{model_name}_output') 
        
      axarr[3].imshow(res0_final1.astype(bool).astype(int), cmap='viridis' )
      axarr[3].set_title(f'{model_name}_prediction') 
        
      axarr[4].imshow(a01,cmap='gray_r')
      axarr[4].plot(vec_layer.T) # gt    
      axarr[4].set_title( f'Vec_layer({thresh} overlaid)') #.set_text
        
      axarr[5].imshow(a01.squeeze(),cmap='gray_r')          
      axarr[5].plot(new_layer_filtered.T) # gt
      axarr[5].set_title( 'Filtered Overlaid prediction') #.set_text
        
      axarr[6].imshow(a01.squeeze(),cmap='gray_r')          
      axarr[6].plot(predict_data['vec_layer'].T) # gt
      axarr[6].set_title( 'Overlaid GT') #.set_text
      
      
            
      # if save_pred:
      #     if not os.path.exists(os.path.join(out_dir,L_folder, model_name)):
      #         os.makedirs(os.path.join(out_dir,L_folder,model_name), exist_ok=True) 
                        
      #     save_path = os.path.join(out_dir,L_folder,model_name,base_name)
      #     out_dict = {} 
      #     out_dict['model_output'] = res0
      #     out_dict['binary_output'] = res0_final1
      #     out_dict['vec_layer'] = vec_layer
      #     out_dict['filtered_vec_layer'] = new_layer_filtered
      #     out_dict['GT_layer'] = predict_data['vec_layer']
      #     savemat(save_path,out_dict)










































# # Predict on the entire test data
# test_pred = model.predict(x_test)
# test_pred = np.argmax(test_pred,axis =3)
# from sklearn.metrics import classification_report
# print( classification_report( y_test.flatten(),test_pred.flatten(), labels=list(range(num_classes)), zero_division=1 ))

# IoU = MeanIoU(num_classes = num_classes)
# IoU.update_state(y_test.flatten(), test_pred.flatten() );
# MIoU = IoU.result().numpy()
# print(f'Mean IoU is {100*MIoU:.2f}%')

# #roc_auc_score(y_test.flatten(), test_pred.flatten(),multi_class='OvR', labels=list(range(30)) )
















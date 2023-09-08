# -*- coding: utf-8 -*-
"""
Originally created on Mon Apr 18 10:16:35 2022
Used for Class Project and not edited afterwards (except for adding Colormap and thickness code)


Now retraining_ Spe 18th_2022

@author: i368o351
"""

from tensorflow.keras import layers
from tensorflow import keras
from keras import backend as K

import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm,colors

import os
import random
from scipy.io import loadmat
from scipy.ndimage import median_filter as sc_med_filt

from keras.metrics import MeanIoU
from sklearn.metrics import roc_auc_score

import glob
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,ReduceLROnPlateau
#import segmentation_models as sm
from datetime import datetime


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
tf.keras.mixed_precision.set_global_policy('mixed_float16')



use_wandb = True
time_stamp = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')

if use_wandb:    
    ## WandB config
    import wandb
    from wandb.keras import WandbCallback    
    
    wandb.init( project="my-test-project", entity="ibksolar", name='KerasViT_train_further'+time_stamp, config = {} )
    config = wandb.config
else:
    config ={}

try:
    fname = ipynbname.name()
except:
    fname = os.path.splitext( os.path.basename(__file__) )[0]
finally:
    print ('Could not automatically find file path')
    fname = 'blank'

# ==================LOAD DATA =========================================
#========================================================================

# PATHS
# Path to data
# base_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data'  # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'
base_path = r'K:\Users\cresis\Desktop\Fall_2021\all_block_data\Attention_Train_data\new_trainJuly' 
train_path = os.path.join(base_path,'train_data\*.mat')
val_path = os.path.join(base_path,'val_data1\*.mat')
test_path = os.path.join(base_path,'test_data\*.mat')   

# Hyperparameters
config['Run_Note'] = 'Old Keras Vit train further'
config['batch_size'] = 8


# Training params
config['img_y'] = 416
config['img_x'] = 64

config['img_channels'] = 1
config['weight_decay'] = 0.0001

config['num_classes'] = 30
config['epochs'] = 500
config['learning_rate'] = 5e-4
config['base_path'] = base_path
SEED = 42
AUTO = tf.data.experimental.AUTOTUNE


# =============================================================================
# Function for training data
def read_mat_train(filepath):
    def _read_mat(filepath):
        
        filepath = bytes.decode(filepath.numpy())      
        mat_file = loadmat(filepath)
        echo = tf.cast(mat_file['echo_tmp'], dtype=tf.float64)
        
        echo = tf.expand_dims(echo, axis=-1)
        
        if config['img_channels'] > 1:
            echo = tf.image.grayscale_to_rgb(echo)
        
        layer = tf.cast(mat_file['semantic_seg2'], dtype=tf.float64)
        #layer = tf.cast( tf.cast(mat_file['raster'], dtype=tf.bool), dtype=tf.float64)
        
        # Data Augmentation
        
        
        aug_type = tf.random.uniform((1,1),minval=1, maxval=8,dtype=tf.int64).numpy()
        
        if aug_type == 1:
            echo = tf.experimental.numpy.fliplr(echo)
            layer = tf.experimental.numpy.fliplr(layer)
        
        elif aug_type == 2: # Constant offset
            echo = echo - 0.3
        
        elif aug_type == 3: # Random noise
            echo = echo - tf.random.normal(shape=(1664,256,config['img_channels']),stddev=0.5,dtype=tf.float64)
            echo = tf.clip_by_value(echo, 0, 1)
            
        elif aug_type == 4: # Random brightness
            echo = tf.image.random_brightness(echo,0.2, 123)
            echo = tf.clip_by_value(echo, 0, 1)
            
        elif aug_type == 5: # Random contrast
            echo = tf.image.random_contrast(echo,0.2, 123)
            echo = tf.clip_by_value(echo, 0, 1)                
        
        elif aug_type == 6: # Random hue
            echo = tf.image.random_hue(echo,0.2, 123)
            echo = tf.clip_by_value(echo, 0, 1)
            
        elif aug_type == 7: # Random brightness
            echo = tf.image.random_saturation(echo, 0.1, 0.9)
            echo = tf.clip_by_value(echo, 0, 1)
        
        else: #aug_type == 4:
            echo = tf.experimental.numpy.flipud(echo)
            layer = tf.experimental.numpy.flipud(layer)                            
                   
        layer = np.expand_dims(layer, axis=-1)
        #layer = tf.keras.utils.to_categorical(layer, config['num_classes'] )
        shape0 = echo.shape #mat_file['echo_tmp'].shape
        
        return echo,layer,np.asarray(shape0)
    output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double, tf.int64])
    shape = output[2]
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([1664,256,config['img_channels']])
    
    data1 = output[1]   
    data1.set_shape([1664,256,1 ])#,30,config['num_classes']    
    return data0,data1

# =============================================================================
## Function for test and validation dataset    
def read_mat(filepath):
    def _read_mat(filepath):
        
        filepath = bytes.decode(filepath.numpy())      
        mat_file = loadmat(filepath)
        echo = tf.cast(mat_file['echo_tmp'], dtype=tf.float64)
        
        # echo = tf.expand_dims(echo, axis=-1)        
        if config['img_channels'] > 1:
            echo = tf.image.grayscale_to_rgb(echo)
        
        layer = tf.cast(mat_file['semantic_seg2'], dtype=tf.float64)            
        # layer = tf.expand_dims(layer, axis=-1)
        
        layer = tf.keras.utils.to_categorical(layer, config['num_classes'] )
        shape0 = echo.shape #mat_file['echo_tmp'].shape        
        return echo,layer,np.asarray(shape0)     
    output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double, tf.int64])
    shape = output[2]
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([config['img_y'],config['img_x']]) #,config['img_channels']
    
    data1 = output[1]   
    data1.set_shape([config['img_y'],config['img_x'],30 ]) #,30,1  
    return data0,data1

train_ds = tf.data.Dataset.list_files(train_path,shuffle=True) #'*.mat'
train_ds = train_ds.map(read_mat,num_parallel_calls=8)
train_ds = train_ds.batch(config['batch_size'],drop_remainder=True).prefetch(AUTO) #.shuffle(buffer_size = 100 * config['batch_size'])

# No augmentation for testing and validation
val_ds = tf.data.Dataset.list_files(val_path,shuffle=True)
val_ds = val_ds.map(read_mat,num_parallel_calls=8)
val_ds = val_ds.batch(config['batch_size'],drop_remainder=True).cache().prefetch(AUTO)

test_ds = tf.data.Dataset.list_files(test_path,shuffle=True)
test_ds = test_ds.map(read_mat,num_parallel_calls=8)
test_ds = test_ds.batch(config['batch_size'],drop_remainder=True).cache().prefetch(AUTO)


train_shape = [ ( tf.shape(item[0]).numpy(),tf.shape(item[1]).numpy() ) for item in train_ds.take(1) ]
train_shape = train_shape[0]

print(f' X_train train shape {train_shape[0]}')
print(f' Training target shape {train_shape[1]}')


# Model hyper-param
# Not needed for train_further
embed_dim = 64 #train_shape.shape[-1]
num_patches = 416 #train_shape.shape[1]
num_heads = 20
dense_dim = 512

transformer_units = [
    embed_dim * 2,
    embed_dim,
]


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches = num_patches, embed_dim = embed_dim, **kwargs):
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

    # Override function to avoid error while saving model
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {               
                "num_patches": num_patches,
            }
        )
        return config


# Losses
alpha = 0.1
gamma = 10

class FocalLoss(tf.losses.Loss):
    """Implements Focal loss"""

    def __init__(self, alpha = alpha, gamma = gamma,**kwargs):
        super(FocalLoss, self).__init__(
            reduction="none", name="RetinaNetFocalLoss"
        )
        self._alpha = tf.cast(alpha,tf.float64)
        self._gamma = tf.cast(gamma, tf.float64)

    def call(self, y_true, y_pred):
        
        y_true = tf.cast(y_true, tf.float64)
        y_pred = tf.cast(y_pred, tf.float64)
        
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred,
        )
        probs = tf.nn.sigmoid(y_pred)
        alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)
    
    # def get_config(self):
    #     config = super().get_config().copy()
    #     config.update({            
    #         'alpha' : self._alpha,
    #         'gamma' : self._gamma,
            
    #         })
    #     return config
custom_loss = FocalLoss(alpha = alpha, gamma =gamma )


 

# Load previous model
model = tf.keras.models.load_model(r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data_old\echo_vit\echo_vit1_Checkpoint05_June_22_1759.h5',custom_objects={'PatchEncoder':PatchEncoder})


# Reset model learning rate
#model.optimizer.learning_rate = 5e-4

opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate']) #, clipnorm=0.5
opt2 = tfa.optimizers.AdamW(weight_decay = 0.001, learning_rate=config['learning_rate'])


config['start_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
logz= f"{config['base_path']}/train_further_EchoViT1/train_further_EchoViT1_{config['start_time']}_logs/"
callbacks = [
   ModelCheckpoint(f"{config['base_path']}/train_further_EchoViT1/train_further_EchoViT1_{config['start_time']}_Checkpoint{time_stamp}.h5", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10, min_lr=5e-8, verbose= 1),
    EarlyStopping(monitor="val_loss", patience=30, verbose=1), 
    TensorBoard(log_dir = logz,histogram_freq = 1,profile_batch = '1,70', embeddings_freq=50),
    WandbCallback()
]



history = model.fit(train_ds, epochs = config['epochs'],validation_data = val_ds, callbacks = callbacks)


if type(loss) is not keras.losses.CategoricalCrossentropy:
    loaded_model = tf.keras.models.load_model(f"{config['base_path']}/train_further_EchoViT1/train_further_EchoViT1_{config['start_time']}_Checkpoint{time_stamp}.h5",
                                   custom_objects={"PatchEncoder":PatchEncoder,"FocalLoss":FocalLoss})
else:
    loaded_model = tf.keras.models.load_model(f"{config['base_path']}/train_further_EchoViT1/train_further_EchoViT1_{config['start_time']}_Checkpoint{time_stamp}.h5",
                                       custom_objects={"PatchEncoder":PatchEncoder})
        


# Save model with proper name
_,acc = model.evaluate(test_ds)
model.save(f"{config['base_path']}/train_further_EchoViT1/train_further_EchoViT1_{config['start_time']}_acc_{acc:.3f}_{time_stamp}.h5")


custom_cm = cm.Blues(np.linspace(0,1,30))
custom_cm = colors.ListedColormap(custom_cm[10:,:-1])


import random

model_val_data_path = os.path.join(base_path,'test_data\*.mat')
model_val_data = glob.glob(model_val_data_path)


# # Visualize result of model prediction for "unseen" echogram during training

batch_idx = random.randint(1,len(model_val_data)) # Pick any of the default batch

for idx in range(1,10):
  predict_data = loadmat(model_val_data[batch_idx+idx])
  a01,a_gt0 = predict_data['echo_tmp'], predict_data['semantic_seg']
  if config['img_channels']  >1:
      a0 = np.stack((a01,)*3,axis=-1)
      res0 = model.predict ( np.expand_dims(a0,axis=0))
  else:
      res0 = model.predict ( np.expand_dims(np.expand_dims(a01,axis=0),axis=3) ) 
      
  res0 = res0.squeeze()
  res0_final = np.argmax(res0,axis=2)
  # res0_final = np.where(res0>0.4,1,0)
  pred0_final = sc_med_filt( sc_med_filt(res0_final,size=3).T, size=3, mode='nearest').T
  
  
  f, axarr = plt.subplots(1,4,figsize=(20,20))

  axarr[0].imshow(a01.squeeze(),cmap='gray_r');
  axarr[0].set_title( f'Echo {os.path.basename(model_val_data[batch_idx])}'); #.set_text
  
  axarr[1].imshow(res0_final, cmap='viridis' );
  axarr[1].set_title('Prediction');
  
  axarr[2].imshow(pred0_final, cmap='viridis') ;
  axarr[2].set_title('Filtered Prediction');

  axarr[3].imshow(a_gt0.squeeze(),cmap='viridis') ;# gt
  axarr[3].set_title( f'Ground truth {os.path.basename(model_val_data[batch_idx])}'); #.set_text




# Predict on the entire test data
test_pred = model.predict(x_test)
test_pred = np.argmax(test_pred,axis =3)

from sklearn.metrics import classification_report
print( classification_report( y_test.flatten(),test_pred.flatten(), labels=list(range(num_classes)), zero_division=1 ))


IoU = MeanIoU(num_classes = num_classes)
IoU.update_state(y_test.flatten(), test_pred.flatten() );
MIoU = IoU.result().numpy()
print(f'Mean IoU is {100*MIoU:.2f}%')

#roc_auc_score(y_test.flatten(), test_pred.flatten(),multi_class='OvR', labels=list(range(30)) )


import itertools

def compute_layer_thickness(model,test_data):
    test_pred = model.predict(test_data)
    test_pred = np.argmax( test_pred, axis = 3)
    test_result = np.zeros((test_pred.shape[0],30,test_pred.shape[-1] ))
    ## Calculate thickness for each block
    
    for block_idx in range(test_pred.shape[0]):
        
        # First filter prediction
        temp1 = sc_med_filt( sc_med_filt(test_pred[block_idx],size=5).T, size=5, mode='nearest').T        
        
        for col_idx in range(temp1.shape[1]):
            uniq_layers = np.unique(temp1[:,col_idx])
            uniq_layers = uniq_layers[uniq_layers>0] # Zero is no layer class
            
            col_thickness = [ len(list(group)) for key, group in itertools.groupby(temp1[:,col_idx] ) if key ]
            
            for layer_idx in range(len(uniq_layers)):
                test_result[block_idx, uniq_layers[layer_idx], col_idx ] = col_thickness[layer_idx]
                
    test_result = np.transpose(test_result,axes= [1,0,2])
    test_result = test_result.reshape((30,-1))
    
    return test_result
                  
            
layer_thickness = compute_layer_thickness(model, x_test)        
        
        
(np.mean(layer_thickness,axis =1)).round()














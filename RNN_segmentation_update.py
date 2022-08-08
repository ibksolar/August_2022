# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:36:06 2022

@author: Ibikunle

No transpose of x_train
Doing transpose and re-transpose in PatchEncoder function

"""

# Imports
from tensorflow.keras import layers
from tensorflow import keras
from keras import backend as K

import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

import os
import random
from scipy.io import loadmat
from scipy.ndimage import median_filter as sc_med_filt

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


## WandB config
# import wandb
# from wandb.keras import WandbCallback
time_stamp = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')

try:
    fname = ipynbname.name()
except:
    fname = os.path.splitext( os.path.basename(__file__) )[0]
finally:
    print ('Could not automatically find file path')
    fname = 'blank'

# wandb.init(project="my-test-project", entity="ibksolar", name=fname+'_'+time_stamp, config={} )
# config = wandb.config

# PATHS
# Path to data
echo_path = r'Y:\ibikunle\Python_Env\final_layers_rowblock15_21'  # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'
echo_path_full = str(echo_path + "\\filtered_image\\image*.mat")

def echo_load(path: str) -> tuple[tf.Tensor,tf.Tensor] :
    """    Function to load echogram and corresponding map
    
        Parameters
        ----------
        img_path : str
            Image (not the mask) location.
        
        Returns
        -------
        dict
            Dictionary mapping an image and its annotation.
    """
    def _echo_load(path):
        matfile = loadmat(path)
        echo = matfile['filtered_img']
        
        label_path = path.replace("filtered_image", "segment_dir")
        matfile2 = loadmat(label_path)
        label = matfile2['semantic_seg']
        return echo,label
    
    return tf.numpy_function(_echo_load, [path], [tf.float32,tf.uint8])

# Vizualize some of  train_ds
def display_sample(display_list):
    """Show side-by-side an input image,
    the ground truth and the prediction.
    """
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

# train_ds = tf.data.Dataset.list_files(echo_path_full, shuffle=True, seed= SEED).map(echo_load,num_parallel_calls=16 )
# train_ds_final = train_ds.shuffle(buffer_size = BUFFER_SIZE, seed= SEED).repeat().batch(BATCH_SIZE).prefetch(AUTO)    
# for image, mask in train_ds_final.take(1):
#     sample_image, sample_mask = image, mask
# display_sample([sample_image[0], sample_mask[0]])


img_paths = glob.glob(echo_path_full)

seg_type = 'segment'

def Create_ListDataLoader( img_paths,seg_type = 'segment' ):
    x_train, y_train = [],[]
    
    for iter,path in enumerate(img_paths):
         x_new = loadmat(path)
         x_new = x_new['filtered_img']
         x_new[np.isnan(x_new)] = 0  
         # x_new = np.expand_dims(x_new,2)
         x_train.append(x_new)          
          
         if seg_type == 'segment':
             path2 = path.replace('filtered_image','segment_dir')
             y_new = loadmat(path2.replace('_dec','_dec_segment'))
             y_new = y_new['semantic_seg']                       
         else:
              path2 = path.replace('filtered_image','raster_dir')
              y_new = loadmat(path2.replace('_dec','_dec_raster'))
              y_new = y_new['raster']         
         
         y_new[np.isnan(y_new)] = 0      
         # y_new = np.expand_dims(y_new,2)
         y_train.append(y_new)
    

    # Out of loop    
    x_train = np.array(x_train)
    #x_train = np.expand_dims(x_train,3)
    
    if seg_type == 'segment':
        y_train = np.asarray(y_train)
    else:    
        y_train = ( np.array(y_train,dtype=bool ) ).astype('float16')
    
    #y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    print(f'y_train shape after categorical{y_train.shape}')
    
    return (x_train,y_train)

train_samples = round(0.768* len(img_paths)  ) #1000 # 1000        
val_samples = 250 # 500
test_samples = len(img_paths) - train_samples - val_samples

random.Random(1337).shuffle(img_paths)

train_input_img_paths = img_paths[:train_samples] # input_img_paths[:-val_samples]
val_input_img_paths = img_paths[train_samples:train_samples+val_samples] # input_img_paths[-val_samples:]
test_img_paths = img_paths[train_samples+val_samples+1:]

x_train,y_train = Create_ListDataLoader(train_input_img_paths, seg_type = seg_type )
x_val,y_val = Create_ListDataLoader(val_input_img_paths, seg_type = seg_type)
x_test,y_test = Create_ListDataLoader(test_img_paths, seg_type = seg_type)

# Transpose data for LSTM
#x_train = tf.transpose(x_train,perm=[0,2,1])
#x_val = tf.transpose(x_val,perm=[0,2,1])


num_classes = int(np.max(y_train)+1) #np.max(y_train)+

y_train_1hot = tf.keras.utils.to_categorical(y_train, num_classes)
y_val_1hot = tf.keras.utils.to_categorical(y_val, num_classes)
y_test_1hot = tf.keras.utils.to_categorical(y_test, num_classes)

# Create tf.data.Dataset
BATCH_SIZE = 4
BUFFER_SIZE = 1000
SEED = 42
AUTO = tf.data.AUTOTUNE

#  Default hyper-param
config={}
config['epochs'] = 700
config['batch_size'] = BATCH_SIZE
config['learning_rate'] = 1e-4

# Model hyper-param
embed_dim = x_train.shape[1]
transformer_embed_dim = x_train.shape[-1]
num_patches = x_train.shape[-1]
num_heads = 10
dense_dim = 512
#num_classes = np.max(y_train)


train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train_1hot))
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTO) #.cache()

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val_1hot))
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTO) #.cache()

test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test_1hot))
test_ds = test_ds.batch(BATCH_SIZE).cache().prefetch(AUTO)


#########################################################################
# Losses
alpha = 0.3
gamma = 4.
class FocalLoss(tf.losses.Loss):
    """Implements Focal loss"""

    def __init__(self, alpha=alpha, gamma=gamma,**kwargs):
        super(FocalLoss, self).__init__(
            reduction="none", name="RetinaNetFocalLoss",**kwargs
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
custom_loss = FocalLoss(alpha = 0.3, gamma =4. )

smooth = 3

def dice_coef(y_true,y_pred):
    
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum( y_true_f * y_pred_f )
    return (2. * intersection +smooth)/ ( K.sum(y_true_f) + K.sum(y_pred_f) + smooth )

def dice_coef_loss(y_true,y_pred):
    return 1 - dice_coef(y_true,y_pred)

#custom_dice_loss = dice_coef_loss()


#######################################################################
# Create classes for model
class TransformerEncoder(layers.Layer):
    def __init__(self, transformer_embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.transformer_embed_dim = transformer_embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        
        self.attention1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=transformer_embed_dim, dropout=0.2)
        
        self.attention2 = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=transformer_embed_dim)
        
        self.dense_proj = tf.keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(dense_dim, activation="relu"),
             layers.Dense(dense_dim//2, activation="relu"),
             layers.Dense(64),]
        )
        self.layernorm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        
        # Repeat attention K times        
        attention_output1 = inputs # New copy to be over-written
        for _ in range(2):
            attention_output1 = self.attention1(
            attention_output1, attention_output1, attention_mask=mask)            
        
        # Repeat attention K times 
        attention_output = self.layernorm_1(attention_output1) # New copy to be over-written
        for _ in range(2):
            attention_output = self.attention2(
            attention_output, attention_output, attention_mask=mask)
        
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "transformer_embed_dim": self.transformer_embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches = num_patches, embed_dim = embed_dim,**kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=embed_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=embed_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        patch_t = tf.transpose(patch,perm=[0,2,1])
        #embed1 = tf.transpose(embed1,perm=[1,0]) # Reshape to match patch dimensions
        encoded = self.projection(patch_t) + self.position_embedding(positions)
        encoded = tf.transpose(encoded,perm=[0,2,1])
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            #"embed_dim": self.embed_dim,            
        })
        return config


strategy = tf.distribute.MirroredStrategy( cross_device_ops=tf.distribute.HierarchicalCopyAllReduce() )
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
with strategy.scope():
    
    # Create Model
    input_shape = (x_train.shape[1:])
    echo_inputs = tf.keras.Input(shape=input_shape, dtype='float64')    

    #echo_inputs_t = tf.transpose(echo_inputs,perm=[0,2,1])
    
    x = PatchEncoder(num_patches, embed_dim)(echo_inputs)
    
    #x = tf.transpose(x,perm=[0,2,1])
        
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = tf.cast(x,dtype="float16") # Keep a residual branch
    
    for _ in range(10):
        x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
        x = x + res
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    #x = tf.transpose(x,perm=[0,2,1])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    x = tf.expand_dims(x, axis=-1)
    x = layers.Conv2D(num_classes*5, (1,1), activation="relu",padding="same")(x)     
    x = layers.Conv2D(num_classes*5, (1,1), activation="relu",padding="same")(x) #, kernel_initializer='he_normal'
    x = layers.Conv2D(num_classes, (1,1), activation="relu",padding="same")(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x) 
    output = layers.Conv2D(num_classes, (1,1), activation="softmax",padding="same")(x)  
    
    model = tf.keras.Model(echo_inputs, output) 
    
    opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate']) #, clipnorm=0.5
    opt2 = tfa.optimizers.AdamW(weight_decay = 0.001, learning_rate=config['learning_rate'])
    # Call backs

    config['base_path2'] = os.path.abspath(r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\PulsedTrainTest').replace(os.sep,'/')
    config['start_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
    logz= f"{config['base_path2']}/{fname}/{config['start_time']}_logs/"
    callbacks = [
       ModelCheckpoint(f"{config['base_path2']}//{fname}//{os.environ['COMPUTERNAME']}_Checkpoint{time_stamp}.h5", save_best_only=True, monitor="val_loss"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=0.000005, verbose= 1),
        EarlyStopping(monitor="val_loss", patience=70, verbose=1), 
        #TensorBoard(log_dir = logz,histogram_freq = 1,profile_batch = '1,70', embeddings_freq=50),
        #WandbCallback()
    ]
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.CategoricalCrossentropy(), #tf.keras.losses.CategoricalCrossentropy(), #custom_loss, dice_coef_loss tf.keras.losses.CategoricalCrossentropy()
                  metrics=['accuracy'])

history = model.fit(train_ds, epochs = config['epochs'],validation_data = val_ds, callbacks = callbacks)

# Load best checkpoint
model = tf.keras.models.load_model(f"{config['base_path2']}//{fname}//{os.environ['COMPUTERNAME']}_Checkpoint{time_stamp}.h5",
                                   custom_objects={'TransformerEncoder':TransformerEncoder,"PatchEncoder":PatchEncoder,"FocalLoss":FocalLoss})

# Save model with proper name
_,acc = model.evaluate(x_test,y_test_1hot)
model.save(f"{config['base_path2']}//{fname}//{os.environ['COMPUTERNAME']}_AttentionSegmentation_{acc:.3f}_{time_stamp}_{seg_type}.h5")


import random

for _ in range(10):
    
    idx = random.randint(1,len(x_test)-1) # Pick any of the default batch    
    test_data,test_gt = x_test[idx],y_test[idx]
       
    
    # Predict
    pred0 = model.predict ( np.expand_dims(test_data,axis=0) ).squeeze()    
    pred0_raw = np.argmax(pred0, axis =2 )
    pred0_final = sc_med_filt( sc_med_filt(pred0_raw,size=3).T, size=3, mode='nearest').T
    
    # plot image
    f, axarr = plt.subplots(1,4,figsize=(15,15))

    axarr[0].imshow(test_data,cmap='gray_r')
    axarr[0].set_title( f'Echo {os.path.basename(test_img_paths[idx])}') #.set_text
  
    # Plot ground truth  
    axarr[1].imshow(test_gt,cmap=cm.get_cmap('viridis', 30)) # gt
    axarr[1].set_title( 'GT') #.set_text

    # Plot prediction 
    axarr[2].imshow(pred0_raw,cmap=cm.get_cmap('viridis', 30)) # gt
    axarr[2].set_title(' Raw Prediction') #.set_text

    # Plot prediction 
    axarr[2].imshow(pred0_final,cmap=cm.get_cmap('viridis', 30)) # gt
    axarr[2].set_title('Corrected Prediction') #.set_text





























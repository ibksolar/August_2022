# -*- coding: utf-8 -*-
"""
Created on Thu May  5 08:15:51 2022
Seq2Seq_Echogram_Layer_Tracking

@author: i368o351
"""

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


# PATHS
# Path to data
echo_path = r'Y:\ibikunle\Python_Env\final_layers_rowblock15_21'  # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'
echo_path_full = str(echo_path + "\\filtered_image\\image*.mat")


try:
    fname = ipynbname.name()
except:
    fname = os.path.splitext( os.path.basename(__file__) )[0]
finally:
    print ('Could not automatically find file path')
    fname = 'blank'

# Prepare data
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
              
         # Add leading(START) and ending (END) placeholders to target
         append_shp = ( y_new.shape[0], 1 )
         y_new = np.hstack( ( np.zeros(append_shp),y_new, np.ones(append_shp) )   )
              
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

img_paths = glob.glob(echo_path_full)

seg_type = 'segment'

train_samples = round(0.75* len(img_paths)  ) #1000 # 1000        
val_samples = 250 # 500
test_samples = len(img_paths) - train_samples - val_samples

random.Random(1337).shuffle(img_paths)

train_input_img_paths = img_paths[:train_samples] # input_img_paths[:-val_samples]
val_input_img_paths = img_paths[train_samples:train_samples+val_samples] # input_img_paths[-val_samples:]
test_img_paths = img_paths[train_samples+val_samples+1:]

x_train,y_train = Create_ListDataLoader(train_input_img_paths, seg_type = seg_type )
x_val,y_val = Create_ListDataLoader(val_input_img_paths, seg_type = seg_type)
x_test,y_test = Create_ListDataLoader(test_img_paths, seg_type = seg_type)

num_classes = int(np.max(y_train)+1) #np.max(y_train)
y_train_1hot = tf.keras.utils.to_categorical(y_train, num_classes)
y_val_1hot = tf.keras.utils.to_categorical(y_val, num_classes)
y_test_1hot = tf.keras.utils.to_categorical(y_test, num_classes)


BATCH_SIZE = 4
BUFFER_SIZE = 1000
SEED = 42
AUTO = tf.data.AUTOTUNE

# Transpose data for LSTM
x_train_t = tf.transpose(x_train,perm=[0,2,1])
x_val_t = tf.transpose(x_val,perm=[0,2,1])
x_test_t = tf.transpose(x_test,perm=[0,2,1])



train_ds = tf.data.Dataset.from_tensor_slices( ({'echo':x_train,'label':y_train[:,:,:-2]}, y_train_1hot[:,:,1:-1,:]) )
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTO) #.cache()

val_ds = tf.data.Dataset.from_tensor_slices( ({'echo':x_val,'label':y_val[:,:,:-2]}, y_val_1hot[:,:,1:-1,:]) )
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTO) #.cache()

test_ds = tf.data.Dataset.from_tensor_slices( ({'echo':x_test,'label':y_test[:,:,:-2]}, y_test_1hot[:,:,1:-1,:]) )
test_ds = test_ds.batch(BATCH_SIZE).cache().prefetch(AUTO)

# Check dimension of tf.data
for inputs, targets in train_ds.take(1):
    print(f"inputs['echo'].shape: {inputs['echo'].shape}")
    print(f"inputs['label'].shape: {inputs['label'].shape}")
    print(f"targets.shape: {targets.shape}")
    

# Model hyper-param
sequence_length = x_train.shape[-1]
embed_dim = x_train.shape[-1]

num_patches = x_train.shape[1]
num_heads = 20
dense_dim = 512   
mlp_head_units = [2048, 1024, 512, 64]
transformer_layers = 20  

# Classes for Model

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config

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
    
    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)
    
    # Override function to avoid error while saving model
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {               
                "num_patches": num_patches,
            }
        )
        return config

    
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(
            inputs, inputs, ) #attention_mask=mask
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config  
    

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1),
             tf.constant([1, 1], dtype=tf.int32)], axis=0)
        return tf.tile(mask, mult)

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(
                mask[:, tf.newaxis, :], dtype="int32")
            #padding_mask = tf.minimum(padding_mask, causal_mask)
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=causal_mask) #Pass the causal mask to the first attention layer, which performs self-attention over the target sequence.
        attention_output_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(
            query=attention_output_1,
            value=encoder_outputs,
            key=encoder_outputs,
            
        ) #attention_mask=padding_mask
        attention_output_2 = self.layernorm_2(
            attention_output_1 + attention_output_2)
        proj_output = self.dense_proj(attention_output_2)
        return self.layernorm_3(attention_output_2 + proj_output)    
    
     
    
# Model
# Legacy
# embed_dim = 256
# dense_dim = 2048
# num_heads = 8

#mlp_head_units = [2048, 1024, 512, 64] 

# Model params
config={}
config['epochs'] = 700
config['batch_size'] = BATCH_SIZE
config['learning_rate'] = 1e-4

# Project inputs
dense_proj = tf.keras.Sequential()
for units in mlp_head_units:
    dense_proj.add(layers.Dense(units, activation='relu', use_bias=True) )

input_shape = (x_train.shape[1:])
#decoder_shape = y_train[:,:,1:-1].shape

encoder_inputs = keras.Input(shape= (input_shape), name="echo")
x = PatchEncoder(num_patches, embed_dim)(encoder_inputs)

encoder_outputs = dense_proj(x)

for _ in range(transformer_layers):
    encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(encoder_outputs)

#encoder_outputs = dense_proj(encoder_outputs)
encoder_outputs = layers.LayerNormalization(epsilon=1e-6)(encoder_outputs)

decoder_inputs = keras.Input(shape=(None,), name="label") # Decoder input  , dtype="int64"
#x = PatchEncoder(num_patches, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, dense_dim, num_heads)(decoder_inputs, encoder_outputs)

representation = layers.LayerNormalization(epsilon=1e-6)(x)

#representation = dense_proj(representation)
#x = layers.Dropout(0.5)(x)

representation= tf.expand_dims(representation, axis=-1)  

representation = layers.Conv2D(num_classes*5, (7,5), activation=tf.nn.gelu, padding="same")(representation)
representation = layers.Conv2D(num_classes*5, (1,1), activation='relu', padding="same")(representation)
decoder_outputs = layers.Conv2D(num_classes, (1,1), activation="softmax", padding="same")(representation)

transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)    
    
    
    
opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate']) #, clipnorm=0.5
opt2 = tfa.optimizers.AdamW(weight_decay = 0.001, learning_rate=config['learning_rate'])

config['base_path2'] = os.path.abspath(r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\PulsedTrainTest').replace(os.sep,'/')
config['start_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
logz= f"{config['base_path2']}/{fname}/{config['start_time']}_logs/"
callbacks = [
    ModelCheckpoint(f"{config['base_path2']}//{fname}//{os.environ['COMPUTERNAME']}_Checkpoint{time_stamp}.h5", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=0.000005, verbose= 1),
    EarlyStopping(monitor="val_loss", patience=30, verbose=1), 
    #TensorBoard(log_dir = logz,histogram_freq = 1,profile_batch = '1,70', embeddings_freq=50),
    #WandbCallback()
]

loss = tf.keras.losses.CategoricalCrossentropy()
transformer.compile(optimizer=opt,
          loss= loss, #tf.keras.losses.CategoricalCrossentropy(), #custom_loss, dice_coef_loss tf.keras.losses.CategoricalCrossentropy()
          metrics=['accuracy']) #,tf.keras.metrics.MeanIoU(num_classes, name="MeanIoU")

history = transformer.fit(train_ds, epochs = config['epochs'],validation_data = val_ds, callbacks = callbacks)
    


# ============================================================================ #
if 0:
    x1,y1 = x_test[1], y_test[1]
    x1 = np.expand_dims(x1,axis=0)
    transformer([x1,np.zeros(416,1)])    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
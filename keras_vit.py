# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 10:16:35 2022
Used for Class Project and not edited afterwards (except for adding Colormap and thickness code)

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

# Create tf.data.Dataset
BATCH_SIZE = 4
BUFFER_SIZE = 1000
SEED = 42
AUTO = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train_1hot))
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTO) #.cache()

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val_1hot))
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTO) #.cache()

test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test_1hot))
test_ds = test_ds.batch(BATCH_SIZE).cache().prefetch(AUTO)


# Training params
config={}
config['epochs'] = 700
config['batch_size'] = BATCH_SIZE
config['learning_rate'] = 1e-3

# Model hyper-param
embed_dim = x_train.shape[-1]
num_patches = x_train.shape[1]
num_heads = 20
dense_dim = 512

transformer_units = [
    embed_dim * 2,
    embed_dim,
]

transformer_layers = 20
mlp_head_units = [2048, 1024, 512, 64]  # Size of the dense layers

input_shape = (x_train.shape[1:])



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


# def create_vit_object_detector(
#     input_shape,   
#     num_patches,
#     embed_dim,
#     num_heads,
#     transformer_units,
#     transformer_layers,
#     mlp_head_units,
# ):
    
inputs = layers.Input(shape=input_shape)

# Encode patches
encoded_patches = PatchEncoder(num_patches, embed_dim)(inputs)

# Project inputs
dense_proj = tf.keras.Sequential()
for units in mlp_head_units:
    dense_proj.add(layers.Dense(units, activation='relu', use_bias=False) )
    
#encoded_patches = dense_proj(encoded_patches)

# Create multiple layers of the Transformer block.
for _ in range(transformer_layers):
    # Layer normalization 1.
    x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    # Create a multi-head attention layer.
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim, dropout=0.1
    )(x1, x1)
    # Skip connection 1.
    x2 = layers.Add()([attention_output, encoded_patches])
    # Layer normalization 2.
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    
    # Skip connection 2.
    encoded_patches = layers.Add()([x3, x2])


# Project Transformer output     
representation = dense_proj(encoded_patches)

# Create a [batch_size, embed_dim] tensor.
representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

representation= tf.expand_dims(representation, axis=-1)  

representation = layers.Conv2D(num_classes*5, (7,5), activation=tf.nn.gelu, padding="same")(representation)
representation = layers.Conv2D(num_classes*5, (1,1), activation='relu', padding="same")(representation)
representation = layers.Conv2D(num_classes, (1,1), activation="relu", padding="same")(representation)    
# layers.Dropout(0.3)(representation)
output = layers.Conv2D(num_classes, (1,1), activation="softmax",padding="same")(representation)
 

# return Keras model.
model = keras.Model(inputs=inputs, outputs=output)

#model =  create_vit_object_detector(input_shape,num_patches,embed_dim,num_heads,transformer_units,transformer_layers,mlp_head_units,)

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
model.compile(optimizer=opt,
          loss= loss, #tf.keras.losses.CategoricalCrossentropy(), #custom_loss, dice_coef_loss tf.keras.losses.CategoricalCrossentropy()
          metrics=['accuracy']) #,tf.keras.metrics.MeanIoU(num_classes, name="MeanIoU")

history = model.fit(train_ds, epochs = config['epochs'],validation_data = val_ds, callbacks = callbacks)


if type(loss) is not keras.losses.CategoricalCrossentropy:
    loaded_model = tf.keras.models.load_model(f"{config['base_path2']}//{fname}//{os.environ['COMPUTERNAME']}_Checkpoint{time_stamp}.h5",
                                   custom_objects={"PatchEncoder":PatchEncoder,"FocalLoss":FocalLoss})
else:
    loaded_model = tf.keras.models.load_model(f"{config['base_path2']}//{fname}//{os.environ['COMPUTERNAME']}_Checkpoint{time_stamp}.h5",
                                       custom_objects={"PatchEncoder":PatchEncoder})
        


# Save model with proper name
_,acc = model.evaluate(x_test,y_test_1hot)
model.save(f"{config['base_path2']}//{fname}//{os.environ['COMPUTERNAME']}_AttentionSegmentation_{acc:.3f}_{time_stamp}_{seg_type}_custome_loss.h5")


custom_cm = cm.Blues(np.linspace(0,1,30))
custom_cm = colors.ListedColormap(custom_cm[10:,:-1])


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

    axarr[0].imshow(test_data, cmap='gray_r')
    axarr[0].set_title( f'Echo {os.path.basename(test_img_paths[idx])}') #.set_text
  
    # Plot ground truth  
    axarr[1].imshow(test_gt, cmap = custom_cm) # gt
    axarr[1].set_title( 'GT') #.set_text

    # Plot prediction 
    axarr[2].imshow(pred0_raw, cmap = custom_cm) # gt
    axarr[2].set_title(' Raw Prediction') #.set_text

    # Plot prediction 
    axarr[3].imshow(pred0_final, cmap = custom_cm) # gt
    axarr[3].set_title('Corrected Prediction') #.set_text



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














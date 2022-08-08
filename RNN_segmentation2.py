# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 10:03:47 2022

@author: i368o351
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:36:06 2022

@author: Ibikunle
"""

# Imports
from tensorflow.keras import layers
from tensorflow import keras

import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import os
import random
from scipy.io import loadmat
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


# WandB config
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

seg_type = 'raster'

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

x_train,y_train = Create_ListDataLoader(train_input_img_paths, seg_type = seg_type)
x_val,y_val = Create_ListDataLoader(val_input_img_paths, seg_type = seg_type)
x_test,y_test = Create_ListDataLoader(test_img_paths, seg_type = seg_type)

# Transpose data for LSTM
x_train = tf.transpose(x_train,perm=[0,2,1])
x_val = tf.transpose(x_val,perm=[0,2,1])

num_classes = int(np.max(y_train)+1)

y_train_1hot = tf.keras.utils.to_categorical(y_train, num_classes)
y_val_1hot = tf.keras.utils.to_categorical(y_val, num_classes)



# Create tf.data.Dataset
BATCH_SIZE = 2
BUFFER_SIZE = 1000
SEED = 42
AUTO = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train_1hot))
train_ds = train_ds.batch(BATCH_SIZE).cache().prefetch(AUTO)

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val_1hot))
val_ds = val_ds.batch(BATCH_SIZE).cache().prefetch(AUTO)

# test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test_1hot))
# test_ds = test_ds.batch(BATCH_SIZE).cache().prefetch(AUTO)
# Create Model



#  Default hyper-param
config={}
config['epochs'] = 500
config['batch_size'] = 128
config['learning_rate'] = 3e-4 

# Model hyper-param
config['head_size'] = head_size = x_train.shape[-1] 
config['mlp_units']= BATCH_SIZE  
config['mlp_dropout']= 0.5     
config['dropout']= 0.15         
config['r_dropout'] = r_dropout = 0.5


# Custom Focal loss
def focal_loss(gamma=2., alpha=4.):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        #y_true = tf.convert_to_tensor(y_true, tf.float16)
        #y_pred = tf.convert_to_tensor(y_pred, tf.float16)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed


class RetinaNetFocalLoss(tf.losses.Loss):
    """Implements Focal loss"""

    def __init__(self, alpha, gamma):
        super(RetinaNetFocalLoss, self).__init__(
            reduction="none", name="RetinaNetFocalLoss"
        )
        self._alpha = tf.cast(alpha,tf.float64)
        self._gamma = tf.cast(gamma, tf.float64)

    def call(self, y_true, y_pred):
        
        y_true = tf.cast(y_true, tf.float64)
        y_pred = tf.cast(y_pred, tf.float64)

        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        probs = tf.nn.sigmoid(y_pred)
        alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

custom_loss = RetinaNetFocalLoss(alpha =1., gamma =4. )

# Create Model
input_shape = (x_train.shape[1:])
echo_inputs = tf.keras.Input(shape=input_shape)

x = layers.LSTM(head_size,recurrent_dropout= r_dropout, return_sequences=True)(echo_inputs) #input_shape=(x.shape[1:]),
x = layers.LSTM(head_size,recurrent_dropout= r_dropout, return_sequences=True)(x)
x = layers.LSTM(head_size,recurrent_dropout= r_dropout, return_sequences=True)(x)
x = layers.LSTM(head_size,recurrent_dropout= r_dropout, return_sequences=True)(x)
x = layers.LayerNormalization()(x)
x = tf.transpose(x,perm=[0,2,1])
x = tf.expand_dims(x, axis=-1)

x = layers.Conv2D(num_classes*5, (1,1), activation="relu",padding="same")(x)
x = layers.Conv2D(num_classes*5, (1,1), activation="relu",padding="same")(x)
 
output = layers.Conv2D(num_classes, (1,1), activation="softmax",padding="same")(x) 

model = tf.keras.Model(echo_inputs, output) 

# Call backs

config['base_path2'] = os.path.abspath(r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\PulsedTrainTest').replace(os.sep,'/')
config['start_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
logz= f"{config['base_path2']}/{fname}/{config['start_time']}_logs/"
callbacks = [
    ModelCheckpoint(f"{config['base_path2']}//{fname}//RowBlockLSTM_ReRun_Checkpoint{time_stamp}.h5", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=20, min_lr=0.00005, verbose= 1),
    EarlyStopping(monitor="val_loss", patience=30, verbose=1), 
    #TensorBoard(log_dir = logz,histogram_freq = 1,profile_batch = '1,70', embeddings_freq=50),
    #WandbCallback()
]

opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'],clipnorm=0.5)
model.compile(optimizer=opt,
              loss=custom_loss,#tf.keras.losses.CategoricalCrossentropy(),custom_loss
              metrics=['accuracy'])

print(f'training start time {config["start_time"]}')

history = model.fit(train_ds, epochs = config['epochs'],validation_data = val_ds, callbacks = callbacks)

config['end_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H_%M')
print(f'End time {config["end_time"]}')



model.save(f'..//all_block_data//PulsedTrainTest//LSTM_Segmentation//{time_stamp}.h5')
#model = tf.keras.models.load_model(f"{config['base_path2']}//{fname}//RowBlockLSTM_ReRun_Checkpoint{time_stamp}.h5")


import random

for idx in range(10,15):
    
    idx = random.randint(1,len(x_test)) # Pick any of the default batch    
    a,gt = x_test[idx],y_test[idx]
       
    
    # Predict
    a0 = model.predict ( np.expand_dims(np.transpose(a),axis=0) )
    a0 = a0.squeeze()
    a0_final = np.argmax(a0, axis =2 )
    
    # plot image
    f, axarr = plt.subplots(1,3,figsize=(15,15))

    axarr[0].imshow(a,cmap='gray_r')
    axarr[0].set_title( f'Echo {os.path.basename(test_img_paths[idx])}') #.set_text
  
    # Plot ground truth  
    axarr[1].imshow(gt,cmap=cm.get_cmap('viridis', 30)) # gt
    axarr[1].set_title( 'GT') #.set_text

    # Plot prediction 
    axarr[2].imshow(a0_final,cmap=cm.get_cmap('viridis', 30)) # gt
    axarr[2].set_title( 'Prediction') #.set_text


























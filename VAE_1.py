# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 15:52:53 2023

@author: i368o351
"""

# This is wrong for now but will be modified

from tensorflow.keras import layers
from tensorflow import keras
#from keras import backend as K
# import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm,colors

import os
import random
from scipy.io import loadmat
from scipy.ndimage import median_filter as sc_med_filt
# import tensorflow_probability as tfp
# from keras.metrics import MeanIoU
# from sklearn.metrics import roc_auc_score

import glob
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,ReduceLROnPlateau
from datetime import datetime


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


model_name = 'VQ_VAE'

use_wandb = True
time_stamp = '24th_April_23_1415' #datetime.strftime( datetime.now(),'%d_%B_%y_%H%M') #'13_December_22_2204'

if use_wandb:    
    ## WandB config
    import wandb
    from wandb.keras import WandbCallback    
    
    wandb.init( project="my-test-project", entity="ibksolar", name= model_name + '_' + time_stamp,config ={})
    config = wandb.config
else:
    config ={}


# PATHS
# Path to data
base_path = r'U:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data'  # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'
# train_aug = os.path.join(base_path,'augmented_plus_train_data\*.mat')
# val_path = os.path.join(base_path,'val_data\*.mat')
test_path = os.path.join(base_path,'test_data\*.mat')   

train_path = os.path.join(base_path,'larger_unsuper_data\good_ones\*.png')

# Create tf.data.Dataset
config['Run_Note'] = 'LastWeekApril_try_VAE'
config['batch_size'] = 4

config['img_y'] = 1664 #1664 , 416
config['img_x'] = 256 #256, 64
config['dropout_rate'] = 0.4

# Training params
#config={}
config['img_channels'] = 1

config['num_classes'] = 1
config['epochs'] = 150
config['learning_rate'] = 1e-3
config['base_path'] = base_path
SEED = 42
AUTO = tf.data.experimental.AUTOTUNE


input_shape = (config['img_y'], config['img_x'], config['img_channels'])


latent_dim = 64
num_embeddings = 128





class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )
        
    
    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices
    
    def get_config(self):
        config = super().get_config()
        config.update( { 
            "embedding_dim": self.embedding_dim,
            "num_embeddings": self.num_embeddings,
            "beta":self.beta
            })


# Encoder and Decoder
used_dtype = tf.float32

def get_encoder(latent_dim=16):
    ResNet_50_model = sm.Unet(backbone_name='resnet50', encoder_weights='imagenet', encoder_freeze=True)    
    encoder_inputs = keras.Input(shape=input_shape ) #(28,28,1)
    in1 = layers.Conv2D(3, (3, 3), activation='relu', padding='same', dtype =used_dtype )(encoder_inputs)
    
    in1 = ResNet_50_model(in1)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(in1) #encoder_inputs
    x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(256, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(512, 3, activation="relu", strides=2, padding="same")(x)
    encoder_outputs = layers.Conv2D(latent_dim, 1, padding="same")(x)
    return keras.Model(encoder_inputs, encoder_outputs, name="encoder")


def get_decoder(latent_dim=16):
    latent_inputs = keras.Input(shape=get_encoder(latent_dim).output.shape[1:])
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(
        latent_inputs
    )
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, padding="same")(x)
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")



# Combined VQ_VAE
def get_vqvae(latent_dim=16, num_embeddings=64):
    vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
    encoder = get_encoder(latent_dim)
    decoder = get_decoder(latent_dim)
    inputs = keras.Input(shape= input_shape ) #(28,28,1)
    encoder_outputs = encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs)
    reconstructions = decoder(quantized_latents)
    return keras.Model(inputs, reconstructions, name="vq_vae")


get_vqvae().summary()

# ==========================================================================


# Wrapping up the training loop inside VQVAETrainer
class VQVAETrainer(keras.models.Model):
    def __init__(self, train_variance, latent_dim=32, num_embeddings=128, **kwargs):
        super().__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.vqvae = get_vqvae(self.latent_dim, self.num_embeddings)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            )
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }



# =============================================================================
## Function for creating dataloader

def read_mat(filepath):
    def _read_mat(filepath):
        
        filepath = bytes.decode(filepath.numpy())      
        mat_file = loadmat(filepath)
        echo = tf.cast(mat_file['echo_tmp']) #, dtype=tf.float64
        
        echo = tf.expand_dims(echo, axis=-1)
        
        if config['img_channels']> 1:            
            echo = tf.image.grayscale_to_rgb(echo)         
        
        shape0 = echo.shape        
        return echo,np.asarray(shape0)     
    
    output = tf.py_function(_read_mat,[filepath],[tf.float32,tf.int32]) #,tf.double
    shape = output[1]
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([config['img_y'],config['img_x'], config['img_channels'] ])    
  
    return data0


def _read_png(filepath):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_png(img)
    img = img/255
    return img

train_ds = tf.data.Dataset.list_files(train_path,shuffle=True) #'*.mat'
train_ds = train_ds.map(_read_png,num_parallel_calls=8)
train_ds = train_ds.batch(config['batch_size'],drop_remainder=True).prefetch(AUTO) #.shuffle(buffer_size = 100 * config['batch_size'])

# test_ds = tf.data.Dataset.list_files(test_path,shuffle=True) #'*.mat'
# test_ds = test_ds.map(_read_png,num_parallel_calls=8)
# test_ds = test_ds.batch(config['batch_size'],drop_remainder=True).prefetch(AUTO)

# =============================================================================


# (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)
# x_train_scaled = (x_train / 255.0) - 0.5
# x_test_scaled = (x_test / 255.0) - 0.5

# data_variance = np.var(x_train / 255.0) #data_variance = 1 #np.var(train_ds)



config['start_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M') 
logz= f"{config['base_path']}/{model_name}/{config['start_time']}_logs/"
callbacks = [
   ModelCheckpoint(f"{config['base_path']}//{model_name}//{model_name}_Checkpoint{time_stamp}.h5", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.25, patience=10, min_lr=0.00001, verbose= 1),
    EarlyStopping(monitor="val_loss", patience=30, verbose=1), 
    TensorBoard(log_dir = logz,histogram_freq = 1,profile_batch = '1,70', embeddings_freq=50),
    WandbCallback()
]

vqvae_trainer = VQVAETrainer(train_variance = 1, latent_dim= latent_dim, num_embeddings = num_embeddings)
vqvae_trainer.compile(optimizer=keras.optimizers.Adam())

history = vqvae_trainer.fit(train_ds, epochs=100, batch_size=64, callbacks=callbacks) #, callbacks=callbacks


test_data = glob.glob(os.path.join(base_path,'test_data/*.mat'))
trained_vqvae_model = vqvae_trainer.vqvae
idx = np.random.choice(len(test_data), 10)

test_images = [test_data[item] for item in idx]

reconstructions_test = trained_vqvae_model.predict(test_images)









































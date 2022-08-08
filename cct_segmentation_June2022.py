# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 18:57:52 2021

@author: i368o351

NOTES: 11/11/2021
    (i) Previous performance was promising... result saved(51%)
    (ii) Yet to run this version with 150 epochs and SM loss and metrics

"""
# Might need to run this first
#   %env SM_FRAMEWORK=tf.keras


# Imports
from tensorflow.keras import layers
from tensorflow import keras
import random

import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import os
from scipy.io import loadmat
from matplotlib import cm

from datetime import datetime
import segmentation_models as sm
import glob

## WandB config
import wandb
from wandb.keras import WandbCallback
time_stamp = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')

wandb.init( project="my-test-project", entity="ibksolar", name='CCTSegment'+time_stamp,config ={})
config = wandb.config

# PATHS
# Path to data
base_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\new_trainJuly'  # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'
train_path = os.path.join(base_path,'train_data\*.mat')
val_path = os.path.join(base_path,'val_data\*.mat')
test_path = os.path.join(base_path,'test_data\*.mat')   

# Create tf.data.Dataset
config['batch_size'] = 16
config['num_classes'] = 30
config['dropout_rate'] = 0.5
config['learning_rate'] = 1e-3
config['epochs'] = 500
config['weight_decay'] = 0.0001


SEED = 42
AUTO = tf.data.experimental.AUTOTUNE

# =============================================================================
# Function for training data
def read_mat_train(filepath):
    def _read_mat(filepath):
        
        filepath = bytes.decode(filepath.numpy())      
        mat_file = loadmat(filepath)
        echo = tf.cast(mat_file['echo_tmp'], dtype=tf.float64)
        layer = tf.cast(mat_file['semantic_seg'], dtype=tf.float64)
        
        # Data Augmentation
        
        if tf.random.uniform(())> 0.5:
            aug_type = tf.random.uniform((1,1),minval=1, maxval=4,dtype=tf.int64).numpy()
            
            if aug_type == 1:
                echo = tf.experimental.numpy.fliplr(echo)
                layer = tf.experimental.numpy.fliplr(layer)
            
            elif aug_type == 2: # Constant offset
                echo = echo - 0.3
            
            elif aug_type == 3: # Random noise
                echo = echo - tf.random.normal(shape=(416,64),stddev=0.5,dtype=tf.float64)
                
            else: #aug_type == 4:
                echo = tf.experimental.numpy.flipud(echo)
                layer = tf.experimental.numpy.flipud(layer)                            
                   
        
        layer = tf.keras.utils.to_categorical(layer, config['num_classes'] )
        shape0 = mat_file['echo_tmp'].shape
        
        return echo,layer,np.asarray(shape0)
    output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double, tf.int64])
    shape = output[2]
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([416,64])
    
    data1 = output[1]   
    data1.set_shape([416,64,30])   #,30 
    return data0,data1

# =============================================================================
## Function for test and validation dataset    
def read_mat(filepath):
    def _read_mat(filepath):
        
        filepath = bytes.decode(filepath.numpy())      
        mat_file = loadmat(filepath)
        echo = tf.cast(mat_file['echo_tmp'], dtype=tf.float64)
        layer = tf.cast(mat_file['semantic_seg'], dtype=tf.float64)      

        layer = tf.keras.utils.to_categorical(layer, config['num_classes'] )
        shape0 = mat_file['echo_tmp'].shape        
        return echo,layer,np.asarray(shape0)     
    
    output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double, tf.int64])
    shape = output[2]
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([416,64])
    
    data1 = output[1]   
    data1.set_shape([416,64,30])  #,30  
    return data0,data1

train_ds = tf.data.Dataset.list_files(train_path,shuffle=True) #'*.mat'
train_ds = train_ds.map(read_mat_train,num_parallel_calls=8)
train_ds = train_ds.batch(config['batch_size'],drop_remainder=True).prefetch(AUTO) #.shuffle(buffer_size = 100 * config['batch_size'])

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


# Hyperparameters and constants

image_size_y = 416 #224, 416
image_size_x = 64
img_channels = 1

input_shape = (image_size_y, image_size_x, img_channels)     # (32, 32, 3)

positional_emb = True
conv_layers = 2
projection_dim = image_size_x #128

num_heads = 2
transformer_units = [projection_dim, projection_dim ]
transformer_layers = 2
stochastic_depth_rate = 0.1


# Redundant :(
img_size = (image_size_y,image_size_x)

# Losses
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss(gamma=5.0)
total_loss = dice_loss + (2*focal_loss)


## The CCT tokenizer

class CCTTokenizer(layers.Layer):
    def __init__(
        self,
        kernel_size=(416,5), #3
        stride=1,
        padding=1,
        pooling_kernel_size=3,
        pooling_stride=2,
        num_conv_layers=conv_layers,
        num_output_channels=[projection_dim,projection_dim], #[64, 128],
        positional_emb=positional_emb,
        **kwargs,
    ):
        super(CCTTokenizer, self).__init__(**kwargs)
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pooling_kernel_size = pooling_kernel_size
        self.pooling_stride = pooling_stride
        self.num_conv_layers = num_conv_layers
        self.num_output_channels = num_output_channels
        self.positional_emb = positional_emb

        # This is our tokenizer.
        self.conv_model = keras.Sequential()
        for i in range(num_conv_layers):
            self.conv_model.add(
                layers.Conv2D(
                    num_output_channels[i],
                    kernel_size,
                    stride,
                    padding="same",
                    use_bias=False,
                    activation="relu",
                    kernel_initializer="he_normal",
                )
            )
            # self.conv_model.add(layers.ZeroPadding2D(padding))
            # self.conv_model.add(
            #     layers.MaxPool2D(pooling_kernel_size, pooling_stride, "same")
            # )

        self.positional_emb = positional_emb
        
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'pooling_kernel_size': self.pooling_kernel_size,
            'pooling_stride': self.pooling_stride,
            'num_conv_layers': self.num_conv_layers,
            'num_output_channels': self.num_output_channels,
            'positional_emb': self.positional_emb        
        })
        return config

    def call(self, images):
        outputs = self.conv_model(images)
        # After passing the images through our mini-network the spatial dimensions
        # summed to keep the dimensions same as image.
        
        # Legacy code: Delete later
        # reshaped = tf.reshape(
        #     outputs,
        #     (-1, tf.shape(outputs)[1] * tf.shape(outputs)[2], tf.shape(outputs)[-1]),
        # )
        reshaped = tf.reduce_sum(outputs,-1)
        return reshaped

    def positional_embedding(self, image_size):
        # Positional embeddings are optional in CCT. Here, we calculate
        # the number of sequences and initialize an `Embedding` layer to
        # compute the positional embeddings later.
        if self.positional_emb:
            dummy_inputs = tf.ones((1, image_size_y, image_size_x, 1))
            dummy_outputs = self.call(dummy_inputs)
            sequence_length = tf.shape(dummy_outputs)[1]
            projection_dim = tf.shape(dummy_outputs)[-1]

            embed_layer = layers.Embedding(
                input_dim=sequence_length, output_dim=projection_dim
            )
            return embed_layer, sequence_length
        else:
            return None


## Stochastic depth for regularization

# Referred from: github.com:rwightman/pytorch-image-models.
class StochasticDepth(layers.Layer):
    def __init__(self, drop_prop,name=None, **kwargs):
        super(StochasticDepth, self).__init__(name=name)
        self.drop_prop = drop_prop        
        super(StochasticDepth, self).__init__(**kwargs)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({            
            'drop_prop' : self.drop_prop,
            
            })
        return config

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_prop
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


## MLP for the Transformers encoder

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


## Data augmentation

# Note the rescaling layer. These layers have pre-defined inference behavior.
# data_augmentation = keras.Sequential(
#     [   
#         layers.RandomZoom(0.3),
#         layers.RandomWidth(0.3),
#         layers.RandomZoom(0.3), 
#         # layers.experimental.preprocessing.RandomCrop(image_size_y, image_size_x),
#         layers.experimental.preprocessing.RandomFlip("vertical"),
#     ],
#     name="data_augmentation",
# )


data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.Rescaling(scale=1.0 / 255),
        layers.experimental.preprocessing.RandomCrop(image_size_y, image_size_x),
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomFlip("vertical"),
    ],
    name="data_augmentation",
)


## The final CCT model
def create_cct_model(
    image_size=(image_size_y, image_size_x),
    input_shape=input_shape,
    num_heads=num_heads,
    projection_dim=projection_dim,
    transformer_units=transformer_units,
):

    inputs = layers.Input(input_shape)

    # Augment data.
    augmented = data_augmentation(inputs)

    # Encode patches.
    cct_tokenizer = CCTTokenizer()
    encoded_patches = cct_tokenizer(augmented) #augmented  inputs

    # Apply positional embedding.
    if positional_emb:
        pos_embed, seq_length = cct_tokenizer.positional_embedding( (image_size_y, image_size_x) )
        positions = tf.range(start=0, limit=seq_length, delta=1)
        position_embeddings = pos_embed(positions)
        encoded_patches += position_embeddings

    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-5)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        x3 = StochasticDepth(dpr[i])(x3)
        encoded_patches = layers.Add()([x3, x2])

    # Apply sequence pooling.
    representation = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)
    representation = tf.expand_dims(representation, -1)
    # attention_weights = tf.nn.softmax(layers.Dense(1)(representation), axis=1)
    # weighted_representation = tf.matmul(
    #     attention_weights, representation, transpose_a=True
    # )
    # weighted_representation = tf.squeeze(weighted_representation, -2)

    # Segment outputs.
    logits = tf.keras.layers.Conv2D(config['num_classes'], (1, 1) )(representation)  #, activation='softmax'   #layers.Dense(num_classes)(weighted_representation)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits, name='CCT_segmentation')
    return model


# Model training and Evaluation

def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=0.1
        ),
        metrics=['accuracy',total_loss,            
            sm.metrics.iou_score,
            #keras.metrics.CategoricalAccuracy(name="accuracy"),
            #tf.keras.metrics.MeanIoU(num_classes= num_classes)
            #keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = base_path + "/CCTSegment/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    ) 
    
    history = model.fit(
        train_ds,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_data = val_ds,#validation_split=0.1,        
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    # _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    # _, accuracy, top_5_accuracy = model.evaluate(test_gen)
    # print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    # print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history,model


cct_model = create_cct_model()
history,model = run_experiment(cct_model)

time_stamp = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')

_,CCT_acc,_,_ = model.evaluate(test_ds)

model_save_path = f'{base_path}/CCT_segmentation/CCT_Acc{CCT_acc*100: .2f}_Epochs_{config["num_epochs"]}_{time_stamp}.h5'
model.save(model_save_path)

# Attempt loading model to be sure
loaded_model = tf.keras.models.load_model(model_save_path
                                ,custom_objects={'CCTTokenizer':CCTTokenizer,'StochasticDepth':StochasticDepth})




## Visualize result of model prediction for "unseen" echogram during training
model_val_data_path = os.path.join(base_path,'test_data\*.mat')
model_val_data = glob.glob(model_val_data_path)

batch_idx = random.randint(1,len(model_val_data)) # Pick any of the default batch

for idx in range(5):
  predict_data = loadmat(model_val_data[batch_idx+idx])
  a0,a_gt0 = predict_data['echo_tmp'], predict_data['semantic_seg']
  
  #a_gt0 = np.asarray(np.asarray(a_gt0,dtype=np.bool_),dtype=np.float32)
  
  res0 = model.predict (np.expand_dims(np.expand_dims(a0,axis=0), axis=3)) 
  res0 = res0.squeeze()
  res0_final = np.argmax(res0,axis=2)


  f, axarr = plt.subplots(1,3,figsize=(20,20))

  axarr[0].imshow(a0.squeeze(),cmap='gray_r')
  axarr[0].set_title( f'Echo {os.path.basename(model_val_data[batch_idx+idx])}') #.set_text
  
  # axarr[1].imshow(res0, cmap='viridis' )
  axarr[1].imshow(res0_final, cmap='viridis' )
  axarr[1].set_title('Prediction')

  axarr[2].imshow(a_gt0.squeeze(),cmap='viridis') # gt
  axarr[2].set_title( 'Ground truth') #.set_text
  
























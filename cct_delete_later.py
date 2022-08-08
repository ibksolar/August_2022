# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 22:15:17 2021

@author: i368o351
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 18:57:52 2021

@author: i368o351
"""

# Imports
from tensorflow.keras import layers
from tensorflow import keras
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import os
from scipy.io import loadmat
from matplotlib import cm

from datetime import datetime


# Data path
data_folder = '/all_block_data/Dec_Train_block_len_21_011121_2331' #all_block_data/Dec_block_len_21_Train_set_291021_1519' # '/all_block_data/Dec_block_len_45_Train_set_181021_1828/'
base_dir = os.path.join ('Y:\ibikunle\Python_Project\Fall_2021' + data_folder ) 

# Confirm path is right...
print(f'{os.path.isdir(base_dir)}')

input_dir = "/image/"
target_dir = "/segment_dir/"

# Hyperparameters and constants

img_hgt = 416 #224, 416
img_wdt = 64
img_channels = 1

input_shape = (img_hgt, img_wdt, img_channels)     # (32, 32, 3)

positional_emb = True
conv_layers = 2
projection_dim = img_wdt #128

num_heads = 2
transformer_units = [projection_dim, projection_dim ]
transformer_layers = 2
stochastic_depth_rate = 0.1

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 32
num_classes = 30

num_epochs = 80

# Redundant :(
img_size = (img_hgt,img_wdt)
image_size_y, image_size_x = img_size


input_img_paths = sorted( os.listdir (base_dir+ input_dir) ) 
target_img_paths = sorted( os.listdir(base_dir + target_dir) ) 

# Echo_Load_Train_Test function
class Echo_Load_Train_Test(keras.utils.Sequence):
    """Dataset loader to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths,base_dir = base_dir ,input_dir = input_dir,target_dir = target_dir, num_classes = num_classes):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.base_dir = base_dir
        self.input_dir = input_dir
        self.target_dir = target_dir

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        
        # x = np.zeros((self.batch_size,) + self.img_size , dtype="uint8")
        x = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img_path = base_dir + input_dir + path
            img = loadmat(img_path)
            img = img['echo_tmp']
            img[np.isnan(img)] = 0
            
            if np.all(img<=1):
                x[j] = np.expand_dims( img, 2) # Normalize /255
            else:
                x[j] = np.expand_dims( img/255, 2)

        # y = np.zeros((self.batch_size,) + self.img_size , dtype="uint8")    
        y = np.zeros((self.batch_size,) + self.img_size + (img_channels,), dtype="float32")
        for j, path in enumerate(batch_target_img_paths):
            target_path = base_dir + target_dir + path
            target = loadmat(target_path)
            target = target['semantic_seg'] #raster
            target = ( np.array(target) ).astype('int') #,dtype=bool
            y[j] = np.expand_dims( target, 2 )
        y = tf.keras.utils.to_categorical(y, num_classes)
        return x, y 


train_samples = round(0.768* len(input_img_paths)  ) #1000 # 1000        
val_samples = 250 # 500
test_samples = len(input_img_paths) - train_samples - val_samples

random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:train_samples] # input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:train_samples] # target_img_paths[:-val_samples]

val_input_img_paths = input_img_paths[train_samples:train_samples+val_samples+1] # input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[train_samples:train_samples+val_samples+1]

# Instantiate training and testing data
train_gen = Echo_Load_Train_Test(batch_size, img_size, train_input_img_paths, train_target_img_paths)
val_gen = Echo_Load_Train_Test(batch_size, img_size, val_input_img_paths, val_target_img_paths)

if test_samples > 1:
    test_input_img_paths = input_img_paths[-test_samples:] # input_img_paths[-val_samples:]
    test_target_img_paths = target_img_paths[-test_samples:]
    test_gen = Echo_Load_Train_Test(batch_size, img_size, test_input_img_paths, test_target_img_paths)



## The CCT tokenizer

class CCTTokenizer(layers.Layer):
    def __init__(
        self,
        kernel_size=3,
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
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.Rescaling(scale=1.0 / 255),
        layers.experimental.preprocessing.RandomCrop(image_size_y, image_size_x),
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
    logits = tf.keras.layers.Conv2D(num_classes, (1, 1) )(representation)  #, activation='softmax'   #layers.Dense(num_classes)(weighted_representation)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


# Model training and Evaluation

def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(learning_rate=0.0001, weight_decay=0.0001)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=0.2
        ),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            #keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = base_dir + "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    ) 
    
    history = model.fit(
        train_gen,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data = val_gen,#validation_split=0.1,        
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
model.save(base_dir+'/CCT_weight/'+time_stamp+'_CCT_21x15.h5')


bp = tf.keras.models.load_model(base_dir+'/CCT_weight'+time_stamp+'_CCT_21x15.h5'
                                ,custom_objects={'CCTTokenizer':CCTTokenizer,'StochasticDepth':StochasticDepth})

# [( np.argmax(y_train[0]), np.argmax( bp.predict( np.expand_dims( np.expand_dims(x_train[idx], axis=0), axis = -1) ) )) for idx in range(10,20)]
























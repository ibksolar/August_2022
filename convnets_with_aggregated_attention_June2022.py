# -*- coding: utf-8 -*-
"""
Created on 21st June 2022

@author: i368o351
"""

import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from tensorflow.keras import layers
from scipy.io import loadmat 
import mat73
from datetime import datetime
import random
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,ReduceLROnPlateau


import os

# Set seed for reproducibiltiy
# SEED = 42
# keras.utils.set_random_seed(SEED)

import wandb
from wandb.keras import WandbCallback
time_stamp = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')

# wandb.init( project="my-test-project", entity="ibksolar", name='ConvNets_w_Agg_Attention'+time_stamp,config ={})
# config = wandb.config


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


# PATHS
# Path to data
base_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Train_data'  # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'
train_path = os.path.join(base_path,'train_data\*.mat')
# val_path = os.path.join(base_path,'val_data\*.mat')
test_path = os.path.join(base_path,'Test_data\*.mat')   

# Create tf.data.Dataset
config = {}
config['batch_size'] = 16
config['num_classes'] = 30
config['dropout_rate'] = 0.5
config['learning_rate'] = 1e-3
config['epochs'] = 500

# ARCHITECTURE
DIMENSIONS = 64 # 256
SE_RATIO = 8
TRUNK_DEPTH = 2

SEED = 42
AUTO = tf.data.experimental.AUTOTUNE


# =============================================================================
# Function for training data
def read_mat_train(filepath):
    def _read_mat(filepath):
        
        filepath = bytes.decode(filepath.numpy())      
        mat_file = loadmat(filepath)
        echo = tf.cast(mat_file['echo_tmp'], dtype=tf.float64)
        # echo = tf.expand_dims(echo,axis = -1)
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
    data1.set_shape([416,64,30])    
    return data0,data1

# =============================================================================
## Function for test and validation dataset    
def read_mat(filepath):
    def _read_mat(filepath):
        
        filepath = bytes.decode(filepath.numpy())      
        mat_file = loadmat(filepath)
        
        echo = tf.cast(mat_file['echo_tmp'], dtype=tf.float64)                
        
        # Convert Grayscale to "RGB" by just repeating
        echo = tf.expand_dims(echo,axis = -1)
        echo = tf.image.grayscale_to_rgb(echo)        
        
        layer = tf.cast(mat_file['semantic_seg'], dtype=tf.float64)    
        # layer = tf.keras.utils.to_categorical(layer, config['num_classes'] )
        
        shape0 = echo.shape #mat_file['echo_tmp'].shape        
        return echo,layer,np.asarray(shape0)     
    
    output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double, tf.int64])
    shape = output[2]
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([416,64,3])
    
    data1 = output[1]   
    data1.set_shape([416,64]) #,30   
    return data0,data1

train_ds = tf.data.Dataset.list_files(train_path,shuffle=True) #'*.mat'
train_ds = train_ds.map(read_mat,num_parallel_calls=8)
train_ds = train_ds.batch(config['batch_size'],drop_remainder=True).prefetch(AUTO) #.shuffle(buffer_size = 100 * config['batch_size'])

# val_ds = tf.data.Dataset.list_files(val_path,shuffle=True)
# val_ds = val_ds.map(read_mat,num_parallel_calls=8)
# val_ds = val_ds.batch(config['batch_size'],drop_remainder=True).cache().prefetch(AUTO)

test_ds = tf.data.Dataset.list_files(test_path,shuffle=True)
test_ds = test_ds.map(read_mat,num_parallel_calls=8)
test_ds = test_ds.batch(config['batch_size'],drop_remainder=True).cache().prefetch(AUTO)

train_shape = [ ( tf.shape(item[0]).numpy(),tf.shape(item[1]).numpy() ) for item in train_ds.take(1) ]
train_shape = train_shape[0]
print(f' X_train train shape {train_shape[0]}')
print(f' Training target shape {train_shape[1]}')

img_size_y = train_shape[0][1]
img_size_x = train_shape[0][-1]

# Augmentation
def get_preprocessing():
    model = keras.Sequential(
        [layers.Rescaling(1 / 255.0), layers.Resizing(img_size_y, img_size_x),],
        name="preprocessing",
    )
    return model


def get_train_augmentation_model():
    model = keras.Sequential(
        [
            #layers.Rescaling(1 / 255.0),
            #layers.Resizing(INPUT_SHAPE[0] + 20, INPUT_SHAPE[0] + 20),
            #layers.RandomCrop(img_size_y, img_size_x),
            layers.RandomFlip("vertical"),
        ],
        name="train_data_augmentation",
    )
    return model


# Conv Stem
def build_convolutional_stem(dimensions):
    """Build the convolutional stem.

    Args:
        dimensions: The embedding dimension of the patches (d in paper).

    Returs:
        The convolutional stem as a keras seqeuntial
        model.
    """
    config = {
        "kernel_size": (3, 3),
        "strides": (2, 2),
        "activation": tf.nn.gelu,
        "padding": "same",
    }

    convolutional_stem = keras.Sequential(
        [
            layers.Conv2D(filters=dimensions // 2, **config),
            layers.Conv2D(filters=dimensions, **config),
        ],
        name="convolutional_stem",
    )

    return convolutional_stem



# Conv Trunk
class SqueezeExcite(layers.Layer):
    """Applies squeeze and excitation to input feature maps as seen in
    https://arxiv.org/abs/1709.01507.

    Args:
        ratio: The ratio with which the feature map needs to be reduced in
        the reduction phase.

    Inputs:
        Convolutional features.

    Outputs:
        Attention modified feature maps.
    """

    def __init__(self, ratio, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config

    def build(self, input_shape):
        filters = input_shape[-1]
        self.squeeze = layers.GlobalAveragePooling2D(keepdims=True)
        self.reduction = layers.Dense(
            units=filters // self.ratio, activation="relu", use_bias=False,
        )
        self.excite = layers.Dense(units=filters, activation="sigmoid", use_bias=False)
        self.multiply = layers.Multiply()

    def call(self, x):
        shortcut = x
        x = self.squeeze(x)
        x = self.reduction(x)
        x = self.excite(x)
        x = self.multiply([shortcut, x])
        return x


class Trunk(layers.Layer):
    """Convolutional residual trunk as in the https://arxiv.org/abs/2112.13692

    Args:
        depth: Number of trunk residual blocks
        dimensions: Dimnesion of the model (denoted by d in the paper)
        ratio: The Squeeze-Excitation ratio

    Inputs:
        Convolutional features extracted from the conv stem.

    Outputs:
        Flattened patches.
    """

    def __init__(self, depth, dimensions, ratio, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.dimensions = dimensions
        self.depth = depth

    def get_config(self):
        config = super().get_config()
        config.update(
            {"ratio": self.ratio, "dimensions": self.dimensions, "depth": self.depth,}
        )
        return config

    def build(self, input_shape):
        config = {
            "filters": self.dimensions,
            "activation": tf.nn.gelu,
            "padding": "same",
        }

        trunk_block = [
            layers.LayerNormalization(epsilon=1e-6),
            layers.Conv2D(kernel_size=(1, 1), **config),
            layers.Conv2D(kernel_size=(3, 3), **config),
            SqueezeExcite(ratio=self.ratio),
            layers.Conv2D(kernel_size=(1, 1), filters=self.dimensions, padding="same"),
        ]

        self.trunk_blocks = [keras.Sequential(trunk_block) for _ in range(self.depth)]
        self.add = layers.Add()
        self.flatten_spatial = layers.Reshape((-1, self.dimensions))

    def call(self, x):
        # Remember the input.
        shortcut = x
        for trunk_block in self.trunk_blocks:
            output = trunk_block(x)
            shortcut = self.add([output, shortcut])
            x = shortcut
        # Flatten the patches.
        x = self.flatten_spatial(x)
        return x
    
    
# Attention Pooling    
class AttentionPooling(layers.Layer):
    """Applies attention to the patches extracted form the
    trunk with the CLS token.

    Args:
        dimensions: The dimension of the whole architecture.
        num_classes: The number of classes in the dataset.

    Inputs:
        Flattened patches from the trunk.

    Outputs:
        The modifies CLS token.
    """

    def __init__(self, dimensions, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.dimensions = dimensions
        self.num_classes = num_classes
        self.cls = tf.Variable(tf.zeros((1, 1, dimensions)))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dimensions": self.dimensions,
                "num_classes": self.num_classes,
                "cls": self.cls.numpy(),
            }
        )
        return config

    def build(self, input_shape):
        self.attention = layers.MultiHeadAttention(
            num_heads=6, key_dim=self.dimensions, dropout=0.2,
        )
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm3 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = keras.Sequential(
            [
                layers.Dense(units=self.dimensions, activation=tf.nn.gelu),
                layers.Dropout(0.2),
                layers.Dense(units=self.dimensions, activation=tf.nn.gelu),
            ]
        )
        self.dense = layers.Dense(units=self.num_classes)
        self.flatten = layers.Flatten()

    def call(self, x):
        batch_size = tf.shape(x)[0]
        # Expand the class token batch number of times.
        #class_token = tf.repeat(self.cls, repeats=batch_size, axis=0)
        # Concat the input with the trainable class token.
        #x = tf.concat([class_token, x], axis=1)
        # Apply attention to x.
        x1 = self.layer_norm1(x)
        x2, viz_weights = self.attention(
            query=x1, key=x1, value=x1, return_attention_scores=True
        ) #[:, 0:1]
        class_token = x1 + x
        class_token = self.layer_norm2(class_token)
        # class_token = self.flatten(class_token)
        # class_token = self.layer_norm3(class_token)
        class_token = class_token + self.mlp(class_token)
        # Build the logits
        logits = class_token #self.dense(class_token)
        return logits, tf.squeeze(viz_weights)[..., 1:]
    
    
    
    
class PatchConvNet(keras.Model):
    def __init__(
        self,
        stem,
        trunk,
        attention_pooling,
        preprocessing_model,
        train_augmentation_model,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.stem = stem
        self.trunk = trunk
        self.attention_pooling = attention_pooling
        self.train_augmentation_model = train_augmentation_model
        self.preprocessing_model = preprocessing_model

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stem": self.stem,
                "trunk": self.trunk,
                "attention_pooling": self.attention_pooling,
                "train_augmentation_model": self.train_augmentation_model,
                "preprocessing_model": self.preprocessing_model,
            }
        )
        return config

    def _calculate_loss(self, inputs, test=False):
        images, labels = inputs
        # Augment the input images.
        if test:
            augmented_images = self.preprocessing_model(images)
        else:
            augmented_images = self.train_augmentation_model(images)
        # Pass through the stem.
        x = self.stem(augmented_images)
        # Pass through the trunk.
        x = self.trunk(x)
        # Pass through the attention pooling block.
        logits, _ = self.attention_pooling(x)
        # Compute the total loss.
        total_loss = self.compiled_loss(labels, logits)
        return total_loss, logits

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            total_loss, logits = self._calculate_loss(inputs)
        # Apply gradients.
        train_vars = [
            self.stem.trainable_variables,
            self.trunk.trainable_variables,
            self.attention_pooling.trainable_variables,
        ]
        grads = tape.gradient(total_loss, train_vars)
        trainable_variable_list = []
        for (grad, var) in zip(grads, train_vars):
            for g, v in zip(grad, var):
                trainable_variable_list.append((g, v))
        self.optimizer.apply_gradients(trainable_variable_list)
        # Report progress.
        _, labels = inputs
        self.compiled_metrics.update_state(labels, logits)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        total_loss, logits = self._calculate_loss(inputs, test=True)
        # Report progress.
        _, labels = inputs
        self.compiled_metrics.update_state(labels, logits)
        return {m.name: m.result() for m in self.metrics}

    def call(self, images):
        # Augment the input images.
        augmented_images = self.preprocessing_model(images)
        # Pass through the stem.
        x = self.stem(augmented_images)
        # Pass through the trunk.
        x = self.trunk(x)
        # Pass through the attention pooling block.
        logits, viz_weights = self.attention_pooling(x)
        return logits, viz_weights
    
    
    
class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")
        cos_annealed_lr = tf.cos(
            self.pi
            * (tf.cast(step, tf.float32) - self.warmup_steps)
            / float(self.total_steps - self.warmup_steps)
        )
        learning_rate = 0.5 * self.learning_rate_base * (1 + cos_annealed_lr)
        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )


total_steps = int((len(train_ds) / config['batch_size']) * config['epochs'])
warmup_epoch_percentage = 0.15
warmup_steps = int(total_steps * warmup_epoch_percentage)
scheduled_lrs = WarmUpCosine(
    learning_rate_base=config['learning_rate'],
    total_steps=total_steps,
    warmup_learning_rate=0.0,
    warmup_steps=warmup_steps,
)
   
    
# Taking a batch of test inputs to measure model's progress.
test_images, test_labels = next(iter(test_ds))


class TrainMonitor(keras.callbacks.Callback):
    def __init__(self, epoch_interval=None):
        self.epoch_interval = epoch_interval

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_interval and epoch % self.epoch_interval == 4:
            test_augmented_images = self.model.preprocessing_model(test_images)
            # Pass through the stem.
            test_x = self.model.stem(test_augmented_images)
            # Pass through the trunk.
            test_x = self.model.trunk(test_x)
            # Pass through the attention pooling block.
            _, test_viz_weights = self.model.attention_pooling(test_x)
            # Reshape the vizualization weights
            num_patches = tf.shape(test_viz_weights)[-1]
            height = width = int(math.sqrt(num_patches))
            test_viz_weights = layers.Reshape((height, width))(test_viz_weights)
            # Take a random image and its attention weights.
            index = np.random.randint(low=0, high=tf.shape(test_augmented_images)[0])
            selected_image = test_augmented_images[index]
            selected_weight = test_viz_weights[index]
            # Plot the images and the overlayed attention map.
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            ax[0].imshow(selected_image)
            ax[0].set_title(f"Original: {epoch:03d}")
            ax[0].axis("off")
            img = ax[1].imshow(selected_image)
            ax[1].imshow(
                selected_weight, cmap="inferno", alpha=0.6, extent=img.get_extent()
            )
            ax[1].set_title(f"Attended: {epoch:03d}")
            ax[1].axis("off")
            plt.axis("off")
            plt.show()
            plt.close()
            
 
## Training            
train_augmentation_model = get_train_augmentation_model()
preprocessing_model = get_preprocessing()
conv_stem = build_convolutional_stem(dimensions=DIMENSIONS)
conv_trunk = Trunk(depth=TRUNK_DEPTH, dimensions=DIMENSIONS, ratio=SE_RATIO)
attention_pooling = AttentionPooling(dimensions=DIMENSIONS, num_classes=config['num_classes'])

patch_convnet = PatchConvNet(
    stem=conv_stem,
    trunk=conv_trunk,
    attention_pooling=attention_pooling,
    train_augmentation_model=train_augmentation_model,
    preprocessing_model=preprocessing_model,
)

# Assemble the callbacks.
# Model and Checkpoint path
config['start_time'] = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
print(f' Start time {config["start_time"]} ')

config['base_path'] = base_path #os.path.abspath(r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\PulsedTrainTest\ConvNetswAggAttention').replace(os.sep,'/')
#model_save_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\PulsedTrainTest/ConvNetswAggAttention'  
logz= f"{config['base_path']}//echo_vit2//{config['start_time']}_logs/"

train_callbacks = [
   ModelCheckpoint(f"{config['base_path']}//ConvNetswAggAttention//ConvNetswAggAttention_Checkpoint{time_stamp}.h5", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=0.000005, verbose= 1),
    EarlyStopping(monitor="val_loss", patience=30, verbose=1), 
    TensorBoard(log_dir = logz,histogram_freq = 1,profile_batch = '1,70', embeddings_freq=50),
    #WandbCallback()
]


# Get the optimizer.
WEIGHT_DECAY = 0.01

optimizer = tfa.optimizers.AdamW(learning_rate=scheduled_lrs, weight_decay=WEIGHT_DECAY)
# Compile and pretrain the model.
patch_convnet.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)
history = patch_convnet.fit(
    train_ds, epochs=config['epochs'], validation_data=test_ds, callbacks=train_callbacks,
)

# Evaluate the model with the test dataset.
loss, acc_top1, acc_top5 = patch_convnet.evaluate(test_ds)
print(f"Loss: {loss:0.2f}")
print(f"Top 1 test accuracy: {acc_top1*100:0.2f}%")
print(f"Top 5 test accuracy: {acc_top5*100:0.2f}%")
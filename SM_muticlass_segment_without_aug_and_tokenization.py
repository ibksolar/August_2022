# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 11:11:04 2021

@author: i368o351
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 12:29:36 2021

Segmentation_models


% This variant of of MultiClass Attention+Sm segmentation script is without augmentation and embedding
 - The output of the SM is passed directly into MSA module
 
 - TO ADD HERE: 11/11/2021
     (i) Did the script even run?
     (ii) Report performance.
                 

@author: i368o351
"""

# Might need to run this first
#   %env SM_FRAMEWORK=tf.keras

# Imports
from tensorflow.keras import layers
from tensorflow import keras

import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
import random
from scipy.io import loadmat

import segmentation_models as sm
from datetime import datetime


# Data path
data_folder = '/all_block_data/Dec_Train_block_len_21_011121_2331' #all_block_data/Dec_block_len_21_Train_set_291021_1519' # '/all_block_data/Dec_block_len_45_Train_set_181021_1828/'
base_dir = os.path.join ('Y:\ibikunle\Python_Project\Fall_2021' + data_folder ) 

# Confirm path is right...
print(f'{os.path.isdir(base_dir)}')


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
    
#tf.keras.mixed_precision.set_global_policy('mixed_float16')



# input_dir = "/image/"
# target_dir = "/segment_dir/"

# # Hyperparameters and constants
image_size_y = 416 #224, 416
image_size_x = 64
img_channels = 3
num_classes = 30


# input_img_paths = sorted( os.listdir (base_dir+ input_dir) ) 
# target_img_paths = sorted( os.listdir(base_dir + target_dir) ) 

# # Create training and testing data
# train_samples = 800 # 1000        
# val_samples = 300 # 500
# test_samples = len(input_img_paths) - train_samples - val_samples

# random.Random(1337).shuffle(input_img_paths)
# random.Random(1337).shuffle(target_img_paths)
# train_input_img_paths = input_img_paths[:train_samples] # input_img_paths[:-val_samples]
# train_target_img_paths = target_img_paths[:train_samples] # target_img_paths[:-val_samples]

# val_input_img_paths = input_img_paths[train_samples:train_samples+val_samples+1] # input_img_paths[-val_samples:]
# val_target_img_paths = target_img_paths[train_samples:train_samples+val_samples+1]

# if test_samples > 1:
#     test_input_img_paths = input_img_paths[-test_samples:] # input_img_paths[-val_samples:]
#     test_target_img_paths = target_img_paths[-test_samples:]


# ## Loading data function
# # Could use a function/class for this but want to accumulate this to try augmentation instead: 
# # Try creating DataLoader for augmentation later

# x_train, y_train = [],[]
# for iter,path in enumerate(train_input_img_paths):
#   x_new = loadmat(base_dir + input_dir + path)
#   x_new = x_new['echo_tmp']
#   x_new[np.isnan(x_new)] = 0
#   # x_new = np.expand_dims(x_new,2)
#   x_train.append(x_new)

  
#   y_new = loadmat(base_dir + target_dir + path.replace('_dec','_dec_segment')) #_dec_raster
#   y_new = y_new['semantic_seg']  #raster
#   y_new[np.isnan(y_new)] = 0
#   # y_new = ( np.array(y_new,dtype=bool)  ).astype('float32')
#   # y_new = np.expand_dims(y_new,2)
#   y_train.append(y_new)

# x_train = np.array(x_train)
# x_train = np.stack((x_train,)*3, axis=-1)
# # x_train = np.expand_dims(x_train,3)

# y_train = (np.array(y_train)).astype('float32') #,dtype=bool
# y_train = np.expand_dims(y_train,3)
# print( x_train.shape, y_train.shape)


# x_val, y_val = [],[]
# for iter,path in enumerate(val_input_img_paths):
#   x_new = loadmat(base_dir + input_dir + path)
#   x_new = x_new['echo_tmp']
#   x_new[np.isnan(x_new)] = 0
#   x_val.append(x_new)
  
#   y_new = loadmat(base_dir + target_dir + path.replace('_dec','_dec_segment'))
#   y_new = y_new['semantic_seg']
#   y_new[np.isnan(y_new)] = 0
#   y_val.append(y_new)

# x_val = np.array(x_val)
# x_val = np.stack((x_val,)*3, axis=-1)
# # x_val = np.expand_dims(x_val,3)

# y_val = (np.array(y_val)).astype('float32') #,dtype=bool
# y_val = np.expand_dims(y_val,3)


# x_test, y_test = [],[]
# for iter,path in enumerate(test_input_img_paths):
#   x_new = loadmat(base_dir + input_dir + path)
#   x_new = x_new['echo_tmp']
#   x_new[np.isnan(x_new)] = 0
#   x_test.append(x_new)
  
#   y_new = loadmat(base_dir + target_dir + path.replace('_dec','_dec_segment'))
#   y_new = y_new['semantic_seg']
#   y_new[np.isnan(y_new)] = 0
#   y_test.append(y_new)

# x_test = np.array(x_test)
# x_test = np.stack((x_test,)*3, axis=-1)
# # x_test = np.expand_dims(x_test,3)

# y_test = (np.array(y_test)).astype('float32') #,dtype=bool
# y_test = np.expand_dims(y_test,3)
# ## Sanity Check
# image_number = random.randint(0, len(x_train))
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.imshow(x_train[image_number, :,:, 0], cmap='gray')
# plt.subplot(122)
# plt.imshow(np.reshape(y_train[image_number], (image_size_y, image_size_x))) #, cmap='gray'
# plt.show()


# PATHS
# Path to data
base_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Train_data'  # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path ) 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'
train_path = os.path.join(base_path,'train_data\*.mat')
val_path = os.path.join(base_path,'val_data\*.mat')
test_path = os.path.join(base_path,'test_data\*.mat')   

# Create tf.data.Dataset
config = {}
config['batch_size'] = 16
config['num_classes'] = 30
SEED = 42
AUTO = tf.data.experimental.AUTOTUNE


# =============================================================================
# Function for training data
def read_mat_train(filepath):
    def _read_mat(filepath):
        
        filepath = bytes.decode(filepath.numpy())      
        mat_file = loadmat(filepath)
        echo = tf.cast(mat_file['echo_tmp'], dtype=tf.float64)
        
        echo = tf.expand_dims(echo,axis = -1)
        echo = tf.image.grayscale_to_rgb(echo)
        
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
                echo = echo - tf.random.normal(shape=(416,64,3),stddev=0.5,dtype=tf.float64)
                
            else: #aug_type == 4:
                echo = tf.experimental.numpy.flipud(echo)
                layer = tf.experimental.numpy.flipud(layer)                            
                   
        
        # layer = tf.keras.utils.to_categorical(layer, config['num_classes'] )
        shape0 = mat_file['echo_tmp'].shape
        
        return echo,layer,np.asarray(shape0)
    output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double, tf.int64])
    shape = output[2]
    data0 = tf.reshape(output[0], shape)
    data0.set_shape([416,64,3])
    
    data1 = output[1]   
    data1.set_shape([416,64]) #,30    
    return data0,data1

# =============================================================================
## Function for test and validation dataset    
def read_mat(filepath):
    def _read_mat(filepath):
        
        filepath = bytes.decode(filepath.numpy())      
        mat_file = loadmat(filepath)        
        
        echo = tf.cast(mat_file['echo_tmp'], dtype=tf.float64)
        
        echo = tf.expand_dims(echo,axis = -1)
        echo = tf.image.grayscale_to_rgb(echo)        
        
        layer = tf.cast(mat_file['semantic_seg'], dtype=tf.float64)      

        # layer = tf.keras.utils.to_categorical(layer, config['num_classes'] )
        shape0 = echo.shape  #mat_file['echo_tmp'].shape        
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

# Training params
#config={}
config['epochs'] = 700
config['learning_rate'] = 1e-3
config['num_classes']= 30
config['dropout'] = 0.4




BACKBONE = 'resnet50'
preprocess_input1 = sm.get_preprocessing(BACKBONE)

# # preprocess input
# images1=preprocess_input1(x_train)
# print(x_train.shape)


# #New generator with rotation and shear where interpolation that comes with rotation and shear are thresholded in masks. 
# #This gives a binary mask rather than a mask with interpolated values. 
# seed=24
# from keras.preprocessing.image import ImageDataGenerator

# img_data_gen_args = dict(rotation_range=90,
#                      width_shift_range=0.3,
#                      height_shift_range=0.3,
#                      shear_range=0.5,
#                      zoom_range=0.3,
#                      horizontal_flip=True,
#                      vertical_flip=True,
#                      fill_mode='reflect')

# mask_data_gen_args = dict(rotation_range=90,
#                      width_shift_range=0.3,
#                      height_shift_range=0.3,
#                      shear_range=0.5,
#                      zoom_range=0.3,
#                      horizontal_flip=True,
#                      vertical_flip=True,
#                      fill_mode='reflect')
#                      # preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) #Binarize the output again. 

# image_data_generator = ImageDataGenerator(**img_data_gen_args)
# image_data_generator.fit(x_train, augment=True, seed=seed)

# image_generator = image_data_generator.flow(x_train, seed=seed)
# test_img_generator = image_data_generator.flow(x_val, seed=seed)
# val_img_generator = image_data_generator.flow(x_val, seed=seed)

# mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
# mask_data_generator.fit(y_train, augment=True, seed=seed)
# mask_generator = mask_data_generator.flow(y_train, seed=seed)
# val_mask_generator = mask_data_generator.flow(y_val, seed=seed)

# def my_image_mask_generator(image_generator, mask_generator):
#     train_generator = zip(image_generator, mask_generator)
#     for (img, mask) in train_generator:
#         yield (img, mask)

# my_generator = my_image_mask_generator(image_generator, mask_generator)
# validation_datagen = my_image_mask_generator(val_img_generator, val_mask_generator)


## Attention Parameters

input_shape = (image_size_y, image_size_x, img_channels)     # (32, 32, 3)
positional_emb = False
conv_layers = 1
projection_dim = image_size_x #128

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 64
num_classes = 30

num_heads = 2
transformer_units = [projection_dim, num_classes ] #projection_dim
transformer_layers = 2
stochastic_depth_rate = 0.1



num_epochs = 70

img_size = (image_size_y,image_size_x)


## The CCT tokenizer

class CCTTokenizer(layers.Layer):
    def __init__(
        self,
        kernel_size= 3, #(416,5), #
        stride=1,
        padding=1,
        pooling_kernel_size=3,
        pooling_stride=2,
        num_conv_layers=conv_layers,
        num_output_channels=[projection_dim,projection_dim], #[64, 128],
        positional_emb=positional_emb,
        num_classes = num_classes,
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
            dummy_inputs = tf.ones((1, image_size_y, image_size_x,1 )) #num_classes
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
            x = tf.cast(x,dtype=tf.float32)
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

## The final CCT model
def create_cct_model(
    image_size=(image_size_y, image_size_x),
    input_shape=input_shape,
    num_heads=num_heads,
    projection_dim=projection_dim,
    transformer_units=transformer_units,
):

    inputs = layers.Input(input_shape,dtype=tf.float32)
    
    # define model
    sm_model = sm.Unet(BACKBONE, encoder_weights='imagenet') #, classes = num_classes, activation = 'softmax'
    
    # Feed input into SM model
    sm_out = sm_model(inputs)
    
    sm_out2 = layers.Conv2D(1,3,activation='relu',padding='same')(sm_out)
    
    # # Encode patches.
    # cct_tokenizer = CCTTokenizer()
    
    # # Output of SM model fed into CCT Tokenizer
    # encoded_patches = cct_tokenizer(sm_out2) #augmented  inputs

    # # Apply positional embedding.
    # if positional_emb:
    #     pos_embed, seq_length = cct_tokenizer.positional_embedding( (image_size_y, image_size_x) )
    #     positions = tf.range(start=0, limit=seq_length, delta=1)
    #     position_embeddings = pos_embed(positions)
    #     encoded_patches += position_embeddings
    
    # # Normalize encoded patches
    # encoded_patches = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)
    
    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-5)(sm_out2) #sm_out , encoded_patches

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = layers.Add()([attention_output, sm_out2]) #encoded_patches

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-5)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        x3 = StochasticDepth(dpr[i])(x3)
        sm_out2 = layers.Add()([x3, x2])

    # Apply sequence pooling.
    representation = layers.LayerNormalization(epsilon=1e-5)(sm_out2) #encoded_patches
    # representation = tf.expand_dims(representation, -1)

    # Segment outputs.
    logits = tf.keras.layers.Conv2D(num_classes, (1, 1),dtype=tf.float32 )(representation)  #, , activation='softmax'   #layers.Dense(num_classes)(weighted_representation)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits, name='CCT_segmentation')
    
    print(model.summary())
    
    return model


# Model training and Evaluation

def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.0001)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, #label_smoothing=0.1
        ),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.MeanIoU(num_classes= num_classes)
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
        train_ds,  #my_generator,
        batch_size=batch_size,
        epochs=num_epochs,
        steps_per_epoch=50, 
        validation_steps=50,
        validation_data = val_ds, #validation_datagen,#validation_split=0.1,        
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    # _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    # _, accuracy, top_5_accuracy = model.evaluate(test_gen)
    # print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    # print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history,model


# model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])
#Fit the model
#history = model.fit(my_generator, validation_data=validation_datagen, steps_per_epoch=len(X_train) // 16, validation_steps=len(X_train) // 16, epochs=100)
# history = model.fit(my_generator, validation_data=validation_datagen, steps_per_epoch=50, validation_steps=50, epochs=500)

cct_model = create_cct_model()
history,model = run_experiment(cct_model)

time_stamp = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')


#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['iou_score']
#acc = history.history['accuracy']
val_acc = history.history['val_iou_score']
#val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training IOU')
plt.plot(epochs, val_acc, 'r', label='Validation IOU')
plt.title('Training and validation IOU')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.legend()
plt.show()


#IOU
# x_test2 =preprocess_input1(x_test)
# y_pred=model.predict(x_test2)
# y_pred_thresholded = y_pred > 0.5

# intersection = np.logical_and(y_test, y_pred_thresholded)
# union = np.logical_or(y_test, y_pred_thresholded)
# iou_score = np.sum(intersection) / np.sum(union)
# print("IoU socre is: ", iou_score)

# test_img_number = random.randint(0, len(x_test)-1)
# # test_img = x_test[test_img_number]
# # test_img_input=np.expand_dims(test_img, 0)
# # ground_truth=y_test[test_img_number]
# # prediction = model.predict(test_img_input)
# # prediction = prediction[0,:,:,0]

# plt.figure(figsize=(16, 8))
# plt.subplot(231)
# plt.title('Testing Image')
# plt.imshow(test_img[:,:,0], cmap='gray')
# plt.subplot(232)
# plt.title('Testing Label')
# plt.imshow(ground_truth[:,:,0], cmap='gray')
# plt.subplot(233)
# plt.title('Prediction on test image')
# plt.imshow(prediction, cmap='gray')








































































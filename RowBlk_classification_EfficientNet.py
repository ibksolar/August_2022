# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 00:00:30 2021

@author: i368o351

Classification_EfficientNet
"""


# Imports
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.models import Sequential

import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import os
import mat73
from scipy.io import loadmat

from tensorflow.keras.applications import EfficientNetB7

from datetime import datetime
import wandb
# from wandb.keras import WandbCallback
# wandb.init(project="Nov2021_Segmentation_Attentions", entity="ibksolar")


# Load dataset

data_folder = '/all_block_data/Old_data/Dec_Train_block_len_21_131121_2213' #all_block_data/Dec_block_len_21_Train_set_291021_1519' # '/all_block_data/Dec_block_len_45_Train_set_181021_1828/'

# base_path = os.path.join (os.getcwd() + data_folder ) 
base_path = os.path.join ('Y:\ibikunle\Python_Project\Fall_2021' + data_folder ) 


# Confirm path is right...
print(f'{os.path.isdir(base_path)}')

raw_data1 = mat73.loadmat(base_path + '/echo_cnn_in_out_jstars.mat')
all_data = raw_data1['echo_cnn_input']
all_target = raw_data1['echo_cnn_target']


# Load old data??
load_old_data = 0

if load_old_data:
    raw_data1 = loadmat('Y:/ibikunle/Python_Env/final_layers_rowblock15_21/filtered_image/new_echo_cnn_in_out_jstars1.mat')
    raw_data2 = loadmat('Y:/ibikunle/Python_Env/final_layers_rowblock15_21/filtered_image/new_echo_cnn_in_out_jstars2.mat')
    raw_data3 = loadmat('Y:/ibikunle/Python_Env/final_layers_rowblock15_21/filtered_image/new_echo_cnn_in_out_jstars3.mat')
# raw_data4 = loadmat('findpeaks_layers_rowblock20/new_echo_cnn_in_out_jstars4.mat')
    d1 = raw_data1['echo_cnn1']
    t1 = raw_data1['echo_target1']
    i1 = raw_data1['echo_idx1']

    d2 = raw_data2['echo_cnn2']
    t2 = raw_data2['echo_target2']
    i2 = raw_data2['echo_idx2']

    d3 = raw_data3['echo_cnn3']
    t3 = raw_data3['echo_target3']
    i3 = raw_data3['echo_idx3']

    all_data = np.concatenate( (d1,d2,d3),axis = 0 )
    all_target = np.concatenate( (t1,t2,t3),axis = 0 )
    all_idx = np.concatenate( (i1,i2,i3),axis = 0 )


row_length = 21 # 45
col_length = 15
num_classes = row_length + 1

all_target[all_target == num_classes] = 0

# Set all nan in the data to zero
nan_idx = np.isnan(all_data).any(axis =-1)
all_target[nan_idx] = 0
all_data[ np.isnan(all_data) ]= 0


input_shape = (row_length, col_length, 3)     # (32, 32, 3)

# Create Train and Test set
x_train, x_test, y_train, y_test = train_test_split(all_data, all_target, test_size=0.2, random_state=1)

# Create validation set
x_test, x_val, y_test, y_val= train_test_split(x_test, y_test, test_size=0.15, random_state=1)

x_train = np.reshape( x_train, (x_train.shape[0],row_length,-1) )
x_train = np.stack((x_train,)*3, axis=-1)

x_test = np.reshape( x_test, (x_test.shape[0],row_length,-1) )
x_test = np.stack((x_test,)*3, axis=-1)

x_val = np.reshape( x_val, (x_val.shape[0],row_length,-1) )
x_val = np.stack((x_val,)*3, axis=-1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")


## Sanity Check
san_check = False
if san_check:
    image_number = random.randint(0, len(x_train))
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(x_train[image_number, :,:, 0], cmap='gray')
    plt.subplot(122)
    plt.imshow(np.reshape(y_train[image_number], (input_shape)), cmap='gray')
    plt.show()

# BACKBONE = 'resnet50'
# preprocess_input1 = sm.get_preprocessing(BACKBONE)
# preprocess input
#images1=preprocess_input1(x_train)



#New generator with rotation and shear where interpolation that comes with rotation and shear are thresholded in masks. 
#This gives a binary mask rather than a mask with interpolated values. 
use_old_augmentation = False
if use_old_augmentation:
    seed=24
    from keras.preprocessing.image import ImageDataGenerator
    
    img_data_gen_args = dict(rotation_range=90,
                         width_shift_range=0.3,
                         height_shift_range=0.3,
                         shear_range=0.5,
                         zoom_range=0.3,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='reflect')
    
    mask_data_gen_args = dict(rotation_range=90,
                         width_shift_range=0.3,
                         height_shift_range=0.3,
                         shear_range=0.5,
                         zoom_range=0.3,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='reflect',
                         preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) #Binarize the output again. 
    
    image_data_generator = ImageDataGenerator(**img_data_gen_args)
    image_data_generator.fit(x_train, augment=True, seed=seed)
    
    image_generator = image_data_generator.flow(x_train, seed=seed)
    test_img_generator = image_data_generator.flow(x_val, seed=seed)
    val_img_generator = image_data_generator.flow(x_val, seed=seed)
    
    def my_image_mask_generator(image_generator, mask_generator):
        train_generator = zip(image_generator, mask_generator)
        for (img, mask) in train_generator:
            yield (img, mask)
    
    my_generator = my_image_mask_generator(image_generator, y_train)
    validation_datagen = my_image_mask_generator(val_img_generator, y_val)
    


img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)


# Define model
inputs = layers.Input(shape = input_shape)
# x = img_augmentation(inputs)
modelB7 = EfficientNetB7(weights='imagenet', input_tensor=inputs, drop_connect_rate=0.2, include_top=False)

modelB7.trainable = False

# Convolution
x = layers.Conv2D(32,3,padding="same",activation='relu')(modelB7.output)
x = layers.Conv2D(64,3,padding="same",activation='relu')(x)

# Output layer
x = layers.Conv2D(128, (1, 1), activation='relu' )(x)  #,    #layers.Dense(num_classes)(weighted_representation)
x = layers.Flatten()(x)
output =  layers.Dense(num_classes, activation='softmax', name="Final_layer") (x)       

# Create the Keras model.
model = tf.keras.Model(inputs=inputs, outputs=output, name='Eff_segmentation_Binary')


model.compile('Adam', loss='categorical_crossentropy', metrics=['accuracy']) #, sm.metrics.iou_score, sm.losses.bce_jaccard_loss
print(model.summary())

#Fit the model
#history = model.fit(my_generator, validation_data=validation_datagen, steps_per_epoch=len(X_train) // 16, validation_steps=len(X_train) // 16, epochs=100)
history = model.fit(x_train,y_train, validation_data=(x_val,y_val), steps_per_epoch=50, validation_steps=50, epochs=50)

## Save Model 
time_stamp = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')
_,Eff_seg_Acc = model.evaluate(x_val)
model_save_path = f'{base_path}/EfficientNet_Bin_segmentation/SM_{Eff_seg_Acc*100: .2f}_{time_stamp}.h5'
model.save(model_save_path)


#Plot the training and validation accuracy and loss at each epoch
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


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training IOU')
plt.plot(epochs, val_acc, 'r', label='Validation IOU')
plt.title('Training and validation IOU')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.legend()
plt.show()


#IOU
y_pred=model.predict(x_test)
y_pred_thresholded = y_pred > 0.5

intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score)

test_img_number = random.randint(0, len(x_test)-1)
test_img = x_test[test_img_number]
test_img_input=np.expand_dims(test_img, 0)
ground_truth=y_test[test_img_number]
prediction = model.predict(test_img_input)
prediction = prediction[0,:,:,0]

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0]) #, cmap='gray'
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction)








































































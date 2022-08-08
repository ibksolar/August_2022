# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 12:29:36 2021

Segmentation_models

@author: i368o351
"""

import numpy as np
from matplotlib import pyplot as plt
import os
import random
from scipy.io import loadmat
from datetime import datetime

# import segmentation_models as sm



# Data path
data_folder = '/all_block_data/Old_data/Dec_Train_block_len_21_131121_2213' #all_block_data/Dec_block_len_21_Train_set_291021_1519' # '/all_block_data/Dec_block_len_45_Train_set_181021_1828/'
base_dir = os.path.join ('Y:\ibikunle\Python_Project\Fall_2021' + data_folder ) 

# Confirm path is right...
print(f'{os.path.isdir(base_dir)}')

input_dir = "/image/"
target_dir = "/raster_dir/" #segment_dir

# Hyperparameters and constants
img_hgt = 416 #224, 416
img_wdt = 64
img_channels = 3

input_img_paths = sorted( os.listdir (base_dir+ input_dir) ) 
target_img_paths = sorted( os.listdir(base_dir + target_dir) ) 

# Create training and testing data
train_samples = 800 # 1000        
val_samples = 300 # 500
test_samples = len(input_img_paths) - train_samples - val_samples

random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:train_samples] # input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:train_samples] # target_img_paths[:-val_samples]

val_input_img_paths = input_img_paths[train_samples:train_samples+val_samples+1] # input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[train_samples:train_samples+val_samples+1]

if test_samples > 1:
    test_input_img_paths = input_img_paths[-test_samples:] # input_img_paths[-val_samples:]
    test_target_img_paths = target_img_paths[-test_samples:]


## Loading data function
# Could use a function/class for this but want to accumulate this to try augmentation instead: 
# Try creating DataLoader for augmentation later

x_train, y_train = [],[]
for iter,path in enumerate(train_input_img_paths):
  x_new = loadmat(base_dir + input_dir + path)
  x_new = x_new['echo_tmp']
  x_new[np.isnan(x_new)] = 0
  # x_new = np.expand_dims(x_new,2)
  x_train.append(x_new)

  
  y_new = loadmat(base_dir + target_dir + path.replace('_dec','_dec_raster')) #_dec_raster
  y_new = y_new['raster']  #raster
  y_new[np.isnan(y_new)] = 0
  # y_new = ( np.array(y_new,dtype=bool)  ).astype('float32')
  # y_new = np.expand_dims(y_new,2)
  y_train.append(y_new)

x_train = np.array(x_train)
x_train = np.stack((x_train,)*3, axis=-1)
# x_train = np.expand_dims(x_train,3)

y_train = (np.array(y_train,dtype=bool)).astype('float32')
y_train = np.expand_dims(y_train,3)
print( x_train.shape, y_train.shape)


x_val, y_val = [],[]
for iter,path in enumerate(val_input_img_paths):
  x_new = loadmat(base_dir + input_dir + path)
  x_new = x_new['echo_tmp']
  x_new[np.isnan(x_new)] = 0
  x_val.append(x_new)
  
  y_new = loadmat(base_dir + target_dir + path.replace('_dec','_dec_raster'))
  y_new = y_new['raster']
  y_new[np.isnan(y_new)] = 0
  y_val.append(y_new)

x_val = np.array(x_val)
x_val = np.stack((x_val,)*3, axis=-1)
# x_val = np.expand_dims(x_val,3)

y_val = (np.array(y_val,dtype=bool)).astype('float32')
y_val = np.expand_dims(y_val,3)


x_test, y_test = [],[]
for iter,path in enumerate(test_input_img_paths):
  x_new = loadmat(base_dir + input_dir + path)
  x_new = x_new['echo_tmp']
  x_new[np.isnan(x_new)] = 0
  x_test.append(x_new)
  
  y_new = loadmat(base_dir + target_dir + path.replace('_dec','_dec_raster'))
  y_new = y_new['raster']
  y_new[np.isnan(y_new)] = 0
  y_test.append(y_new)

x_test = np.array(x_test)
x_test = np.stack((x_test,)*3, axis=-1)
# x_test = np.expand_dims(x_test,3)

y_test = (np.array(y_test,dtype=bool)).astype('float32')
y_test = np.expand_dims(y_test,3)


## Sanity Check
image_number = random.randint(0, len(x_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(x_train[image_number, :,:, 0], cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (img_hgt, img_wdt)), cmap='gray')
plt.show()

BACKBONE = 'resnet50'
preprocess_input1 = sm.get_preprocessing(BACKBONE)

# preprocess input
images1=preprocess_input1(x_train)
print(x_train.shape)


#New generator with rotation and shear where interpolation that comes with rotation and shear are thresholded in masks. 
#This gives a binary mask rather than a mask with interpolated values. 
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

mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
mask_data_generator.fit(y_train, augment=True, seed=seed)
mask_generator = mask_data_generator.flow(y_train, seed=seed)
val_mask_generator = mask_data_generator.flow(y_val, seed=seed)

def my_image_mask_generator(image_generator, mask_generator):
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)

my_generator = my_image_mask_generator(image_generator, mask_generator)
validation_datagen = my_image_mask_generator(val_img_generator, val_mask_generator)



# define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet')
model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])
print(model.summary())

#Fit the model
#history = model.fit(my_generator, validation_data=validation_datagen, steps_per_epoch=len(X_train) // 16, validation_steps=len(X_train) // 16, epochs=100)
history = model.fit(my_generator, validation_data=validation_datagen, steps_per_epoch=50, validation_steps=50, epochs=50)

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



time_stamp = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')

_,SM_seg_Acc = model.evaluate(x_val)

model_save_path = f'{base_dir}/SM_segmentation/SM_{SM_seg_Acc*100: .2f}_{time_stamp}.h5'
# model_save_path = base_path+'/CCT_weight/RowBlockAcc_'+CCT_RowBlock_Acc+time_stamp+'_CCT_21x15.h5'
model.save(model_save_path)




































































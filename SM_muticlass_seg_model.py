# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 12:29:36 2021

Segmentation_models
%env SM_FRAMEWORK=tf.keras
@author: i368o351
"""
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
import random
from scipy.io import loadmat
from tensorflow import keras

import segmentation_models as sm

from datetime import datetime
import wandb
from wandb.keras import WandbCallback
wandb.init(project="Nov2021_Segmentation_Attentions", entity="ibksolar")


# Data path
data_folder = '/all_block_data/Old_data/Dec_Train_block_len_21_131121_2213' #all_block_data/Dec_block_len_21_Train_set_291021_1519' # '/all_block_data/Dec_block_len_45_Train_set_181021_1828/'
base_dir = os.path.join ('Y:\ibikunle\Python_Project\Fall_2021' + data_folder ) 

# Confirm path is right...
print(f'{os.path.isdir(base_dir)}')

input_dir = "/image/"
target_dir = "/segment_dir/"

# Hyperparameters and constants
img_hgt = 416 #224, 416
img_wdt = 64
img_channels = 3
num_classes = 30
num_epochs = 2 # Check steps_per_epochs to choose 

img_size = (img_hgt,img_wdt )

## Wandb.config configuration
wandb.config = {
   "base_dir":base_dir,
   "img_hgt":img_hgt, "img_wdt":img_wdt,
   "img_channels":img_channels, "learning_rate": "NA",
   "weight_decay":"NA",
  "batch_size": "NA", "epochs": num_epochs,   
  
}

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


# ===========================================================================================
#               Function
# ===========================================================================================
    
def Create_ListDataLoader( img_paths, base_dir = base_dir, input_dir = input_dir, target_dir = target_dir):
    x_train, y_train = [],[]
    
    for iter,path in enumerate( img_paths ):
                 
      x_new = loadmat(base_dir + input_dir + path)
      x_new = x_new['echo_tmp']
      x_new[np.isnan(x_new)] = 0  
      # x_new = np.expand_dims(x_new,2)
      x_train.append(x_new)   
      
      y_new = loadmat(base_dir + target_dir + path.replace('_dec','_dec_segment'))
      y_new = y_new['semantic_seg']
      y_new[np.isnan(y_new)] = 0
      y_new = y_new[:img_hgt,:img_wdt]  
      # y_new = np.expand_dims(y_new,2)
      y_train.append(y_new)
    
    x_train = np.array(x_train)
    
    x_train = np.expand_dims(x_train,3)
    
    y_train = np.array(y_train)    
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    print(f'y_train shape after categorical{y_train.shape}')
    
    return (x_train,y_train)
# ===========================================================================================  

    
    
# ===========================================================================================
#              Class Echo_Load_Train_Test
# ===========================================================================================

# Echo_Load_Train_Test function
class Echo_Load_Train_Test(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths,base_dir = base_dir ,input_dir = input_dir,target_dir = target_dir, num_classes = num_classes):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.base_dir = base_dir
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.num_classes = num_classes

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        
        # x = np.zeros((self.batch_size,) + self.img_size , dtype="uint8")
        x = np.zeros((self.batch_size,) + self.img_size , dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img_path = base_dir + input_dir + path
            img = loadmat(img_path)
            img = img['echo_tmp']
            img[np.isnan(img)] = 0
            
            if np.all(img<=1):
                x[j] = img # np.expand_dims( img, 2) # Normalize /255
            else:
                x[j] = img/255   #np.expand_dims( img/255, 2)
        
        x = np.stack((x,)*3, axis=-1) # SM module requires 3 channels

        # y = np.zeros((self.batch_size,) + self.img_size , dtype="uint8")    
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_target_img_paths):
            target_path = base_dir + target_dir + path
            target = loadmat(target_path)
            target = target['semantic_seg'] # raster
            target[np.isnan(target)] = 0
            target = target[:img_hgt,:img_wdt]
            target = ( np.array(target) ).astype('int') #,dtype=bool                        
            y[j] = np.expand_dims( target, 2 )
        y = tf.keras.utils.to_categorical(y, num_classes)
        return x, y  

# ===========================================================================================

# Instantiate DataLoader
batch_size = 20
use_class = 1
if use_class:    
    train_gen = Echo_Load_Train_Test(batch_size, img_size, train_input_img_paths, train_target_img_paths)
    val_gen = Echo_Load_Train_Test(batch_size, img_size, val_input_img_paths, val_target_img_paths)
    
    if test_samples > 1:
        test_input_img_paths = input_img_paths[-test_samples:] # input_img_paths[-val_samples:]
        test_target_img_paths = target_img_paths[-test_samples:]
        test_gen = Echo_Load_Train_Test(batch_size, img_size, test_input_img_paths, test_target_img_paths)


# Instantiations to train model

backbone1 = 'resnet34'
backbone2 = 'resnet50'

resnet34_preprocess = sm.get_preprocessing(backbone1)
resnet50_preprocess = sm.get_preprocessing(backbone2)

x_train34 = resnet34_preprocess(train_gen)
x_val34 = resnet34_preprocess(val_gen)
x_test34 = resnet34_preprocess(test_gen)

x_train50 = resnet50_preprocess(train_gen)
x_val50 = resnet50_preprocess(val_gen)
x_test50 = resnet50_preprocess(test_gen)



## Sanity Check
san_check = False
if san_check:    
    image_number = random.randint(0, len(x_train))
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(x_train[image_number, :,:, 0], cmap='gray')
    plt.subplot(122)
    plt.imshow(np.reshape(y_train[image_number], (img_hgt, img_wdt))) #, cmap='gray'
    plt.show()




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
                     fill_mode='reflect')
                     # preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) #Binarize the output again. 

# image_data_generator = ImageDataGenerator(**img_data_gen_args)
# image_data_generator.fit(x_train, augment=True, seed=seed)

# image_generator = image_data_generator.flow(x_train, seed=seed)
# test_img_generator = image_data_generator.flow(x_test, seed=seed)
# val_img_generator = image_data_generator.flow(x_val, seed=seed)

# mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
# mask_data_generator.fit(y_train, augment=True, seed=seed)

# mask_generator = mask_data_generator.flow(y_train, seed=seed)
# val_mask_generator = mask_data_generator.flow(y_val, seed=seed)
# test_mask_generator = mask_data_generator.flow(y_test, seed=seed) 

# def my_image_mask_generator(image_generator, mask_generator):
#     train_generator = zip(image_generator, mask_generator)
#     for (img, mask) in train_generator:
#         yield (img, mask)

# my_generator = my_image_mask_generator(image_generator, mask_generator)
# validation_datagen = my_image_mask_generator(val_img_generator, val_mask_generator)
# test_gen = my_image_mask_generator(val_img_generator, val_mask_generator)


# define model
ResNet34_model = sm.Unet(backbone1, encoder_weights='imagenet', classes = num_classes, activation = 'softmax')
ResNet34_model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=['accuracy', sm.metrics.iou_score])
#print(ResNet34_model.summary())

#Fit the model
#history = model.fit(my_generator, validation_data=validation_datagen, steps_per_epoch=len(X_train) // 16, validation_steps=len(X_train) // 16, epochs=100)
#history = model.fit(my_generator, validation_data=validation_datagen, steps_per_epoch=5, validation_steps=5, epochs=5, callbacks=[ WandbCallback() ])

history = ResNet34_model.fit(x_train34, epochs= num_epochs, validation_data= x_val34 )


time_stamp = datetime.strftime( datetime.now(),'%d_%B_%y_%H%M')

_,acc,iou = model.evaluate(test_gen)

model_save_path = f'{base_dir}/SM_multiclass_segmentation/Acc_{acc*100: .2f}_Epoch{num_epochs}_{time_stamp}.h5'

model.save(model_save_path)

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
x_test2 =preprocess_input1(x_test)
y_pred=model.predict(x_test2)
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
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')








































































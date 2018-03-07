#By @Kevin Xu
#kevin28520@gmail.com
#Youtube: https://www.youtube.com/channel/UCVCSn4qQXTDAtGWpWAe4Plw
#
#The aim of this project is to use TensorFlow to process our own data.
#    - input_data.py:  read in data and generate batches
#    - model: build the model architecture
#    - training: train

# I used Ubuntu with Python 3.5, TensorFlow 1.0*, other OS should also be good.
# With current settings, 10000 traing steps needed 50 minutes on my laptop.


# data: cats vs. dogs from Kaggle
# Download link: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
# data size: ~540M

# How to run?
# 1. run the training.py once
# 2. call the run_training() in the console to train the model.

# Note: 
# it is suggested to restart your kenel to train the model multiple times 
#(in order to clear all the variables in the memory)
# Otherwise errors may occur: conv1/weights/biases already exist......


#%%

import tensorflow as tf
import numpy as np
import os
from PIL import Image
import math
import cv2

#%%

# you need to change this to your data directory
train_dir = './database0'

def get_Data(file_dir, ratio, IMG_W = 32, IMG_H = 32, CHANNEL = 3):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    image_list = []
    label_list = []
    fdir = file_dir + '/{}/'
    ffdir = file_dir + '/{}/{}'
    for i in range(0,10):
        for img in os.listdir(fdir.format(i)):
            image_list.append(ffdir.format(i,img))
            label_list.append(i)
    print('There are %d images' % len(image_list))
       
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)   
    
    all_image_list = temp[:, 0]
    all_label_list = temp[:, 1]
    
    n_sample = len(all_label_list)
    n_val = math.ceil(n_sample*ratio) # number of validation samples
    n_train = n_sample - n_val # number of trainning samples
    
    tra_images_path = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images_path = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]
    
    tra_images = openImageList(tra_images_path)
    val_images = openImageList(val_images_path)

    return tra_images, tra_labels, val_images, val_labels


def openImageList(imageList, IMG_W = 32, IMG_H = 32, CHANNEL = 3):
    '''
    Args:
        image: list type
        label: list type
        IMG_W: image width
        IMG_H: image height
        BATCH_SIZE: batch size
        CAPACITY: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [BATCH_SIZE, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [BATCH_SIZE], dtype=tf.int32
    '''
    openedImageList = []
    for imagePath in imageList:
        image = Image.open(imagePath).convert('RGB')
        image = np.array(image) / 255#normalize data
        #print(image)
        w,h,c = image.shape
        if w != IMG_W or h != IMG_H or c != CHANNEL:
            print(imagePath)
            continue
        #image.reshape(IMG_W, IMG_H, CHANNEL)
        #print('image.shape = {}'.format(image.shape))      
        openedImageList.append(image)
    
    output = np.stack(openedImageList, axis=0)
    #print('output.shape = {}'.format(output.shape))  
    
    return output
 
#%% TEST
'''
N_CLASSES = 10
IMG_W = 32  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 32
CHANNEL = 3
BATCH_SIZE = 50
RATIO = 0.2 # take 20% of dataset as validation data 
CAPACITY = 2000
epochs = 200
data_augmentation = False
    
train_dir = './data'
    
train, train_label, val, val_label = get_Data(train_dir, RATIO)
'''
#%%





    

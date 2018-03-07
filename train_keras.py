# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:19:10 2017

@author: hsiaoyuh_wang
"""

import input_train_val_split_keras
import model_keras

import os
#import numpy as np
#import tensorflow as tf
#import tensorflow.contrib.keras.python.keras
from tensorflow.contrib.keras.python.keras.optimizers import Adam
from tensorflow.contrib.keras.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.contrib.keras.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.contrib.keras.python.keras.regularizers import l2
from tensorflow.contrib.keras.python.keras import backend as K
#from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.utils import np_utils
import matplotlib.pyplot as plt

N_CLASSES = 10
IMG_W = 32  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 32
CHANNEL = 3
BATCH_SIZE = 50
RATIO = 0.2 # take 20% of dataset as validation data 
#CAPACITY = 2000
epochs = 100
data_augmentation = False

def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
def train():
    
    train_dir = './data'
    logs_train_dir = './logs/train/'
    #logs_val_dir = './logs/val/'
    
    train, train_label, val, val_label = input_train_val_split_keras.get_Data(train_dir, RATIO)
	
    # Convert class vectors to binary class matrices.
    train_label = np_utils.to_categorical(train_label, N_CLASSES)
    val_label = np_utils.to_categorical(val_label, N_CLASSES)
        
    # Input image dimensions.    
    if K.image_data_format() == 'channels_first':
        input_shape = (3, IMG_W, IMG_H)
    else:
        input_shape = (IMG_W, IMG_H, 3)
    
    model = model_keras.inference(input_shape=input_shape, N_CLASSES=N_CLASSES)
    
    model.compile(loss='categorical_crossentropy',
              #optimizer=Adam(lr = model_keras.lr_schedule(0)),
              optimizer='Adam',
              metrics=['accuracy'])
    
    model.summary()    
    
    # Prepare model model saving directory.
    save_dir = os.path.join(logs_train_dir, 'saved_models')
    #model_name = 'cifar10_%s_model.{epoch:03d}.h5' % 'test'
    model_name = 'weights.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_name_first = 'first_try.h5'
    print(model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    filepath_first = os.path.join(save_dir, model_name_first)
    
    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=True)
    
    lr_scheduler = LearningRateScheduler(model_keras.lr_schedule)
    
    #lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
    #                            cooldown=0,
    #                            patience=5,
    #                            min_lr=0.5e-6)
    
    #callbacks = [checkpoint, lr_reducer, lr_scheduler]
    callbacks = [checkpoint, lr_scheduler]
           
    # Run training, with or without data augmentation.
    if not data_augmentation:
        print('Not using data augmentation.')
        #print('train.shape = {}'.format(train.shape))
        #print('train_label.shape = {}'.format(train_label.shape))
        #print('val.shape = {}'.format(val.shape))
        #print('val_label.shape = {}'.format(val_label.shape))
        train_history = model.fit(train, train_label,
                batch_size=BATCH_SIZE,
                epochs=epochs,
                validation_data=(val, val_label),
                shuffle=True,
                callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False)
    
        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(train)
    
        # Fit the model on the batches generated by datagen.flow().
        train_history = model.fit_generator(datagen.flow(train, train_label, batch_size=BATCH_SIZE),
                            validation_data=(val, val_label),
                            epochs=epochs, verbose=1, workers=4,
                            callbacks=callbacks)
    
    model.save_weights(filepath_first)
    
    # Score trained model.
    scores = model.evaluate(train, train_label, verbose=1)
    print('\r\nTrain loss:', scores[0])
    print('\r\nTrain accuracy:', scores[1])
    
    scores = model.evaluate(val, val_label, verbose=1)
    print('\r\nTest loss:', scores[0])
    print('\r\nTest accuracy:', scores[1])
    
    show_train_history(train_history,'acc','val_acc')

train()









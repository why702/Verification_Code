# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 20:38:24 2017

@author: HsiaoYuh_Wang
"""
import CodeSegment_32
import model
from PIL import Image
import matplotlib.pyplot as plt
import requests
import numpy as np
import tensorflow as tf
import cv2
import os

N_CLASSES = 10
IMG_W = 32
IMG_H = 32
CHANNEL = 3
#BATCH_SIZE = 32
CAPACITY = 2000
MAX_STEP = 20000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001 # with current parameters, it is suggested to use learning rate = 0.0001
dropout = 1


def evaluate_one_image(imagePath):
    '''Test one image against the saved models and parameters
    '''
    logs_train_dir = './logs/train/'
    
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 10
        
        image_path = tf.cast(imagePath, tf.string)    
        image_contents = tf.read_file(image_path)
        image = tf.image.decode_jpeg(image_contents, channels=CHANNEL)    
        image = tf.image.per_image_standardization(image)
        image = tf.image.resize_image_with_crop_or_pad(image, IMG_W, IMG_H)
        image = tf.reshape(image, [1, IMG_W, IMG_H, CHANNEL])
        
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, shape = [1, IMG_W, IMG_H, CHANNEL], name='x-input')
        
        hidden5 = model.inference(
            images = x, 
            IMG_W = IMG_W, 
            IMG_H = IMG_H, 
            keep_prob = dropout, 
            BATCH_SIZE = BATCH_SIZE, 
            N_CLASSES = N_CLASSES, 
            CHANNEL = CHANNEL)
        
        logit = tf.nn.softmax(hidden5)        
                        
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            
            #print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                #print('Loading success, global_step is %s' %global_step)
            else:
                print('No checkpoint file found')
                
            test_images = sess.run(image)
            #print(test_images)
            
            prediction = sess.run(logit, feed_dict = {x: test_images})
            max_index = np.argmax(prediction)
            #print('Result: {0}, possibility: {1}'.format(max_index, prediction[:,max_index] * 100))

    return max_index, prediction[:,max_index] * 100

    
#%%


def evaluate_images(listROI):
    '''Test one image against the saved models and parameters
    '''
    logs_train_dir = './logs/train/'
    resultList, possibilityList = [], []
        
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 10
        
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, shape = [1, IMG_W, IMG_H, CHANNEL], name='x-input')
        
        hidden5 = model.inference(
            images = x, 
            IMG_W = IMG_W, 
            IMG_H = IMG_H, 
            keep_prob = dropout, 
            BATCH_SIZE = BATCH_SIZE, 
            N_CLASSES = N_CLASSES, 
            CHANNEL = CHANNEL)
        
        logit = tf.nn.softmax(hidden5)        
                        
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            
            #print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                #print('Loading success, global_step is %s' %global_step)
            else:
                print('No checkpoint file found')
                
                
            for ROI in listROI:       
                np_ROI = np.asarray(ROI, np.float32)
                tf_ROI = tf.convert_to_tensor(np_ROI, np.float32) 
                image = tf.image.per_image_standardization(tf_ROI)
                image = tf.image.resize_image_with_crop_or_pad(image, IMG_W, IMG_H)
                image = tf.reshape(image, [1, IMG_W, IMG_H, CHANNEL])
        
                test_images = sess.run(image)
                prediction = sess.run(logit, feed_dict = {x: test_images})
                max_index = np.argmax(prediction)
                
                if prediction[0,max_index] * 100 < 90.0:
                    continue
                
                resultList.append(max_index)
                possibilityList.append(prediction[0,max_index] * 100)
                #print('Result: {0}, possibility: {1}'.format(max_index, prediction[:,max_index] * 100))

    return resultList, possibilityList

    
#%%
'''
imagePath = 'test.jpg'
with open(imagePath, 'wb') as fig:
    res = requests.get('http://railway.hinet.net/ImageOut.jsp?pageRandom=0.9013912568365248')
    fig.write(res.content)
    
listROI = CodeSegment_32.segmentCode(imagePath);
#figPathList = CodeSegment_32.segmentCode(imagePath);
#print('There are %d images' % len(figPathList))

resultList, possibilityList = evaluate_images(listROI)

print(resultList)
print(possibilityList)
'''
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:53:35 2017

@author: HsiaoYuh_Wang
"""

import tensorflow as tf
from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras import layers
from tensorflow.contrib.keras.python.keras.applications.imagenet_utils import _obtain_input_shape
from tensorflow.contrib.keras.python.keras.applications.imagenet_utils import decode_predictions  # pylint: disable=unused-import
from tensorflow.contrib.keras.python.keras.applications.imagenet_utils import preprocess_input  # pylint: disable=unused-import
from tensorflow.contrib.keras.python.keras.engine.topology import get_source_inputs
from tensorflow.contrib.keras.python.keras.layers import Activation
from tensorflow.contrib.keras.python.keras.layers import AveragePooling2D
from tensorflow.contrib.keras.python.keras.layers import BatchNormalization
from tensorflow.contrib.keras.python.keras.layers import Conv2D
from tensorflow.contrib.keras.python.keras.layers import Dense
from tensorflow.contrib.keras.python.keras.layers import Flatten
from tensorflow.contrib.keras.python.keras.layers import GlobalAveragePooling2D
from tensorflow.contrib.keras.python.keras.layers import GlobalMaxPooling2D
from tensorflow.contrib.keras.python.keras.layers import Input
from tensorflow.contrib.keras.python.keras.layers import MaxPooling2D
from tensorflow.contrib.keras.python.keras.layers import ZeroPadding2D
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.utils import layer_utils
from tensorflow.contrib.keras.python.keras.utils.data_utils import get_file
from tensorflow.contrib.keras.python.keras.regularizers import l2

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 1800:
        lr *= 0.5e-3
    elif epoch > 1500:
        lr *= 1e-3
    elif epoch > 1000:
        lr *= 1e-2
    elif epoch > 500:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def dn_layer(inputs, name,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
    # Returns
        x (tensor): tensor as input to the next layer
    """
    x = Conv2D(num_filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-4),
                   name = name + '_con')(inputs)
    #x = BatchNormalization(name = name + '_bn')(x)
    x = Activation(activation,
                   name = name + '_act')(x)
    return x

def inference(input_shape, N_CLASSES):
    '''build the model
    args:
        image: image batch, 4D tensor [batch_size, width, height, channels=3], dtype=tf.float32
    return:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''

    inputs = Input(shape=input_shape)
    
    K.set_learning_phase(False)  # all new operations will be in test mode from now on
    
    ## Conv layer 1
    x = dn_layer(inputs = inputs, name='layer1')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='layer1_maxpool')(x)
    
    ## Conv layer 2
    x = dn_layer(inputs = x, num_filters = 32, name='layer2')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='layer2_maxpool')(x)
    
    ## Conv layer 3    
    x = dn_layer(inputs = x, num_filters = 64, name='layer3')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='layer3_maxpool')(x)
    
    x = GlobalAveragePooling2D(name='GlobalAveragePooling2D')(x)
    
    #x = Flatten(name='flatten')(x)
    x = Dense(N_CLASSES, activation='softmax', name='predictions')(x)
        
    model = Model(inputs=inputs, outputs=x)
    return model
    
    





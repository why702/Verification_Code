# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:53:35 2017

@author: HsiaoYuh_Wang
"""
import tensorflow as tf

def inference(images, IMG_W, IMG_H, BATCH_SIZE, N_CLASSES, CHANNEL, keep_prob = 1):
    '''build the model
    args:
        image: image batch, 4D tensor [batch_size, width, height, channels=3], dtype=tf.float32
    return:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''

    # Functions for creating weights and biases
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    # Functions for convolution and pooling functions
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    
    def max_pooling_2x2(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    def avg_pooling_8x8(x):
        return tf.nn.avg_pool(x, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME')
    
    def variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
          mean = tf.reduce_mean(var)
          tf.summary.scalar('mean', mean)
          with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
          tf.summary.scalar('stddev', stddev)
          tf.summary.scalar('max', tf.reduce_max(var))
          tf.summary.scalar('min', tf.reduce_min(var))
          tf.summary.histogram('histogram', var)
    	  

    def nn_layer(input_tensor, weight_dim, bias_dim, layer_name, act1=conv2d, act2=tf.nn.relu):
        """Reusable code for making a simple neural net layer.
    
        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
          # This Variable will hold the state of the weights for the layer
          with tf.name_scope('weights'):
            weights = weight_variable(weight_dim)
            variable_summaries(weights)
          with tf.name_scope('biases'):
            biases = bias_variable(bias_dim)
            variable_summaries(biases)
          with tf.name_scope('Wx_plus_b'):
            preactivate = act1(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
          activations = act2(preactivate, name='activation')
          tf.summary.histogram('activations', activations)
          return activations
      
    
    # Input layer
    #first reshape x to a 4d tensor, 
    #with the second and third dimensions corresponding to image width and height, 
    #and the final dimension corresponding to the number of color channels.
    with tf.name_scope('input_reshape'):    
        image_shaped_input = tf.reshape(images, [-1, IMG_W, IMG_H, CHANNEL])
        tf.summary.image('input', image_shaped_input, 10)
    
	
    ## Conv layer 1
    hidden1 = nn_layer(image_shaped_input, [3, 3, CHANNEL, 16], [16], 'layer1', act1=conv2d, act2=tf.nn.relu)
    x_pool1 = max_pooling_2x2(hidden1)
    
    ## Conv layer 2
    hidden2 = nn_layer(x_pool1, [3, 3, 16, 32], [32], 'layer2', act1=conv2d, act2=tf.nn.relu)
    x_pool2 = max_pooling_2x2(hidden2)
    
    ## Conv layer 3
    hidden3 = nn_layer(x_pool2, [3, 3, 32, 64], [64], 'layer3', act1=conv2d, act2=tf.nn.relu)
    #image size has been reduced to 8x8
    #x_pool3 = max_pooling_2x2(hidden3)
    x_pool3 = avg_pooling_8x8(hidden3)
    
    x_flat = tf.reshape(x_pool3, [-1, 1 * 1 * 64])
    #hidden4 = nn_layer(x_flat, [64, 64], [64], 'layer4', act1=tf.matmul, act2=tf.nn.relu)	
	
    # Regularization with dropout
    with tf.name_scope('dropout'):
    #To reduce overfitting, we will apply dropout before the readout layer. 
    #We create a placeholder for the probability that a neuron's output is kept during dropout. 
    #This allows us to turn dropout on during training, and turn it off during testing. 
        tf.summary.scalar('dropout_keep_probability', keep_prob)
    #TensorFlow's tf.nn.dropout op automatically handles scaling neuron outputs in addition to masking them, 
    #so dropout just works without any additional scaling.
        dropped = tf.nn.dropout(x_flat, keep_prob)
    
    hidden4 = nn_layer(dropped, [64, N_CLASSES], [N_CLASSES], 'layer4', act1=tf.matmul, act2=tf.identity)
	
    
    return hidden4
    
    
                       
                           
#%%
def losses(logits, labels):
    '''Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [BATCH_SIZE, N_CLASSES]
        labels: label tensor, tf.int32, [BATCH_SIZE]
        
    Returns:
        loss tensor of float type
    '''
    #print('logits = '.format(logits))
    #print('labels = '.format(labels))
    with tf.variable_scope('loss') as scope:
        #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss

#%%
def trainning(loss, learning_rate):
    '''Training ops, the Op returned by this function is what must be passed to 
        'sess.run()' call to cause the model to train.        
    Args:
        loss: loss tensor, from losses()        
    Returns:
        train_op: The op for trainning
    '''
    with tf.name_scope('optimizer'):
        train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
    return train_op

#%%
def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [BATCH_SIZE, NUM_CLASSES].
    labels: Labels tensor, int32 - [BATCH_SIZE], with values in the range [0, NUM_CLASSES).
  Returns:
    A scalar int32 tensor with the number of examples (out of BATCH_SIZE)
    that were predicted correctly.
  """
  with tf.variable_scope('accuracy') as scope:
      #correct = tf.nn.in_top_k(logits, labels, 1)
      correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
      accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
      tf.summary.scalar(scope.name+'/accuracy', accuracy)
  return accuracy
        
#%%





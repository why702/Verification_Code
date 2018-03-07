# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:19:10 2017

@author: hsiaoyuh_wang
"""

import input_train_val_split
import model

import os
import numpy as np
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data

N_CLASSES = 10
IMG_W = 32  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 32
CHANNEL = 3
BATCH_SIZE = 50
RATIO = 0.2 # take 20% of dataset as validation data 
CAPACITY = 2000

fake_data = False
max_steps = 30000
learning_rate = 0.001
dropout = 0.5
#log_dir = 'D:/svn/ditSRC/exes/ditSPCdbGen/etc/bin/SPCdb/MNIST/log'
#data_dir = 'D:/svn/ditSRC/exes/ditSPCdbGen/etc/bin/SPCdb/MNIST/MNIST_data'


## Import data
#mnist = input_data.read_data_sets(data_dir,
#                                    one_hot=True,
#                                    fake_data=fake_data)
##mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#
## Take a look the training data
#print('Training.images shape: ', mnist.train.images.shape)
#print('Training.labels shape: ', mnist.train.labels.shape)
#print('Shape of an image: ', mnist.train.images[0].shape)
#print('Example label: ', mnist.train.labels[0])
    
def train():
    
    train_dir = './data'
    logs_train_dir = './logs/train/'
    logs_val_dir = './logs/val/'
    
    train, train_label, val, val_label = input_train_val_split.get_files(train_dir, RATIO)
	
    
    train_batch, train_label_batch = input_train_val_split.get_batch_OneHot(train,
                                                  train_label,
                                                  IMG_W,
                                                  IMG_H,
                                                  BATCH_SIZE, 
                                                  CAPACITY,
                                                  CHANNEL)
            
    val_batch, val_label_batch = input_train_val_split.get_batch_OneHot(val,
                                                  val_label,
                                                  IMG_W,
                                                  IMG_H,
                                                  BATCH_SIZE, 
                                                  CAPACITY,
                                                  CHANNEL)    
    
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, CHANNEL], name='x-input')
        y_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE, N_CLASSES], name='y-input')
        keep_prob = tf.placeholder(tf.float32)
        
    # To run nicely in jupyter notebook
    #sess = tf.InteractiveSession()
    hidden5 = model.inference(
            images = x, 
            IMG_W = IMG_W, 
            IMG_H = IMG_H, 
            keep_prob = keep_prob, 
            BATCH_SIZE = BATCH_SIZE, 
            N_CLASSES = N_CLASSES, 
            CHANNEL = CHANNEL)
    
    # Probabilities - output from model (not the same as logits)
    y = tf.nn.softmax(hidden5)#?
    
    # Loss and optimizer
    with tf.name_scope('cross_entropy'):
        # The raw formulation of cross-entropy,
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=hidden5)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)
   
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
 
    # Setup to test accuracy of model
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            #tf.equal: Returns the truth value of (x == y) element-wise.
            #print('tf.argmax(y, 1) = {}'.format(tf.argmax(y, 1)))
            #print('tf.argmax(y_, 1) = {}'.format(tf.argmax(y_, 1)))
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            #tf.cast: Casts a tensor to a new type.
            #tf.reduce_mean: Computes the mean of elements across dimensions of a tensor.
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        
        # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        test_writer = tf.summary.FileWriter(logs_val_dir)
        
        #tf.global_variables_initializer().run()#?
        ## Initilize all global variables
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess, coord = coord)
        
        #bTrain=True: data is batch of train data
        #bTrain=False: data is all of test data
        def feed_dict(bTrain):
            """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
            if bTrain:
                xs, ys = tra_images, tra_labels#mnist.train.next_batch(100, fake_data=fake_data)
                k = dropout
            else:
                xs, ys = val_images, val_labels
                k = 1.0
            #print('len(xs) = {}'.format(len(xs)))
            #print('len(ys) = {}'.format(len(ys)))
            #print('ys[0] = {}'.format(ys[0]))
            #print('k = {}'.format(k))
            return {x: xs, y_: ys, keep_prob: k}
        
        # Train model
        # Run once to get the model to a good confidence level
        for i in range(max_steps):
            if i % 100 == 0:
                val_images, val_labels = sess.run([val_batch, val_label_batch])
                #print('len(val_labels) = {}'.format(len(val_labels)))
                #print('len(val_labels[0]) = {}'.format(len(val_labels[0])))
                #print('val_labels[0] = {}'.format(val_labels[0]))
                summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
                test_writer.add_summary(summary, i)
                print('Accuracy at step %s: %s' % (i, acc))
                
            if i % 2000 == 0 or (i + 1) == max_steps:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step = i)
                
            else:  # Record train set summaries, and train	            
                tra_images, tra_labels = sess.run([train_batch, train_label_batch])
        		
                if i % 200 == 199:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([merged, train_step],
        							feed_dict=feed_dict(True),
        							options=run_options,
        							run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                    train_writer.add_summary(summary, i)
                    #print('Adding run metadata for', i)
                else:  # Record a summary
                    summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                    train_writer.add_summary(summary, i)
        train_writer.close()
        test_writer.close()			
        
        coord.request_stop()
        coord.join(threads)		  


train()

'''	  
def plot_predictions(image_list, output_probs=False, adversarial=False):
    ''
    Evaluate images against trained model and plot images.
    If adversarial == True, replace middle image title appropriately
    Return probability list if output_probs == True
    ''
    prob = y.eval(feed_dict={x: image_list, keep_prob: 1.0})
    
    pred_list = np.zeros(len(image_list)).astype(int)
    pct_list = np.zeros(len(image_list)).astype(int)
    
    # Setup image grid
    import math
    cols = 3
    rows = math.ceil(image_list.shape[0]/cols)
    fig = plt.figure(1, (12., 12.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(rows, cols),  # creates grid of axes
                     axes_pad=0.5,  # pad between axes in inch.
                     )
    
    # Get probs, images and populate grid
    for i in range(len(prob)):
        pred_list[i] = np.argmax(prob[i]) # for mnist index == classification
        pct_list[i] = prob[i][pred_list[i]] * 100

        image = image_list[i].reshape(28,28)
        grid[i].imshow(image)
        
        grid[i].set_title('Label: {0} \nCertainty: {1}%' \
                          .format(pred_list[i], 
                                  pct_list[i]))
        
        # Only use when plotting original, partial deriv and adversarial images
        if (adversarial) & (i % 3 == 1): 
            grid[i].set_title("Adversarial \nPartial Derivatives")
        
    plt.show()
    
    return prob if output_probs else None

# Get 10 2s [:,2] from top 500 [0:500], nonzero returns tuple, get index[0], then first 10 [0:10]
index_of_2s = np.nonzero(mnist.test.labels[0:500][:,2])[0][0:10]
x_batch = mnist.test.images[index_of_2s]

plot_predictions(x_batch)


## Mostly inspired by:
## https://codewords.recurse.com/issues/five/why-do-neural-networks-think-a-panda-is-a-vulture
#def create_plot_adversarial_images(x_image, y_label, lr=0.1, n_steps=1, output_probs=False):
#    
#    original_image = x_image
#    probs_per_step = []
#    
#    # Calculate loss, derivative and create adversarial image
#    # https://www.tensorflow.org/versions/r0.11/api_docs/python/train/gradient_computation
#    loss =  tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y_conv)
#    deriv = tf.gradients(loss, x)
#    image_adv = tf.stop_gradient(x - tf.sign(deriv)*lr/n_steps)
#    image_adv = tf.clip_by_value(image_adv, 0, 1) # prevents -ve values creating 'real' image
#    
#    for _ in range(n_steps):
#        # Calculate derivative and adversarial image
#        dydx = sess.run(deriv, {x: x_image, keep_prob: 1.0}) # can't seem to access 'deriv' w/o running this
#        x_adv = sess.run(image_adv, {x: x_image, keep_prob: 1.0})
#        
#        # Create darray of 3 images - orig, noise/delta, adversarial
#        x_image = np.reshape(x_adv, (1, 784))
#        img_adv_list = original_image
#        img_adv_list = np.append(img_adv_list, dydx[0], axis=0)
#        img_adv_list = np.append(img_adv_list, x_image, axis=0)
#
#        # Print/plot images and return probabilities
#        probs = plot_predictions(img_adv_list, output_probs=output_probs, adversarial=True)
#        probs_per_step.append(probs) if output_probs else None
#    
#    return probs_per_step
#
## Pick a random 2 image from first 1000 images 
## Create adversarial image and with target label 6
#index_of_2s = np.nonzero(mnist.test.labels[0:1000][:,2])[0]
#rand_index = np.random.randint(0, len(index_of_2s))
#image_norm = mnist.test.images[index_of_2s[rand_index]]
#image_norm = np.reshape(image_norm, (1, 784))
#label_adv = [0,0,0,0,0,0,1,0,0,0] # one hot encoded, adversarial label 6
#
## Plot adversarial images
## Over each step, model certainty changes from 2 to 6
#create_plot_adversarial_images(image_norm, label_adv, lr=0.2, n_steps=5)
'''










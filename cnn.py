# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 21:35:48 2016

@author: liujun
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
#%%
#settings
n_labels = 10
n_valid = 5000
n_batch = 128
n_patch =5 
n_hidden = 64
n_steps = 1001
#%%
data = pd.read_csv('train.csv')
print('data shape = ', data.shape)
print(data.head(2))
#%%
all_images = data.iloc[:,1:].values.astype(np.float32)
all_images = all_images/255.
image_size = 28

def display_img(index):
    img = all_images[index,:].reshape(28,28)
    plt.imshow(img)
    plt.show()

display_img(0)

#%%
all_labels = data.iloc[:,0].values
#encoding labels
encode_labels = (np.arange(n_labels) == all_labels[:,None]).astype(np.float32)
print(encode_labels[0],all_labels[0])
#%%
train_imgs = all_images[n_valid:]
train_imgs.shape = (-1,image_size,image_size,1)
train_labels = encode_labels[n_valid:]

valid_imgs = all_images[:n_valid]
valid_imgs.shape = (-1,image_size,image_size,1)
valid_labels = encode_labels[:n_valid]
#%%
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])
#%%
test_data = pd.read_csv('test.csv')
test_imgs = test_data.values.astype(np.float32) / 255.
test_imgs.shape = (-1,image_size,image_size,1)
n_test = test_imgs.shape[0]

#%%
depth = 16

graph = tf.Graph()
with graph.as_default():
    tf_train_imgs = tf.placeholder(tf.float32, shape=(n_batch, image_size, image_size, 1))
    tf_train_labels = tf.placeholder(tf.float32, shape=(n_batch, n_labels))    
    tf_valid_imgs = tf.constant(valid_imgs)
    tf_valid_labels = tf.constant(valid_labels)
    tf_test_imgs = tf.constant(test_imgs)
    
    layer1_weights = tf.Variable(tf.truncated_normal([n_patch, n_patch, 1, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal([n_patch, n_patch, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    layer3_weights = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, n_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[n_hidden]))
    layer4_weights = tf.Variable(tf.truncated_normal([n_hidden, n_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[n_labels]))
    
    
    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases
        
    logits = model(tf_train_imgs)
    loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_imgs))
    test_prediction = tf.nn.softmax(model(tf_test_imgs))


#%%
with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(n_steps):
        offset = (step * n_batch) % (train_labels.shape[0] - n_batch)
        batch_imgs = train_imgs[offset:(offset + n_batch), :, :, :]
        batch_labels = train_labels[offset:(offset + n_batch), :]
        feed_dict = {tf_train_imgs : batch_imgs, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
    #testdata
    test_labels = np.argmax(test_prediction.eval(),1)
    print(test_labels)
    np.savetxt('submission.csv', 
           np.c_[range(1,n_test+1),test_labels], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')
    

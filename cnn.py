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
n_steps = 200001
drop_out = 0.5
learning_rate = 1e-4
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
test_labels = np.zeros((n_test, n_labels))
#%%
depth = 16

graph = tf.Graph()
with graph.as_default():
    tf_train_imgs = tf.placeholder(tf.float32, shape=(n_batch, image_size, image_size, 1))
    tf_train_labels = tf.placeholder(tf.float32, shape=(n_batch, n_labels))    
    tf_valid_imgs = tf.constant(valid_imgs)
    tf_valid_labels = tf.constant(valid_labels)
    tf_test_imgs = tf.placeholder(tf.float32, shape=(1000, image_size, image_size, 1))
    drpt = tf.placeholder(tf.float32)
    
    layer1_weights = tf.Variable(tf.truncated_normal([n_patch, n_patch, 1, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal([n_patch, n_patch, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    layer3_weights = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, n_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[n_hidden]))
    layer4_weights = tf.Variable(tf.truncated_normal([n_hidden, n_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[n_labels]))
    
    global_step = tf.Variable(0)
    
    def model(data):
        #conv1 n*28*28*1
        conv1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.relu(conv1 + layer1_biases)
        #pool1 n*14*14*1
        pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        #conv2 n*14*14*1
        conv2 = tf.nn.conv2d(pool1, layer2_weights, [1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.relu(conv2 + layer2_biases)
        #pool2 n*7*7*1
        pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        shape = pool2.get_shape().as_list()
        reshape = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        hidden_keep = tf.nn.dropout(hidden, drpt)
        return tf.matmul(hidden_keep, layer4_weights) + layer4_biases
        
    logits = model(tf_train_imgs)
    loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    
    learning_rate = tf.train.exponential_decay(0.1, global_step,
                                           1000, 0.9, staircase=True)
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_imgs))
    test_prediction = tf.nn.softmax(model(tf_test_imgs))


#%%
batch_acc = []
valid_acc = []
steps = []

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(n_steps):
        offset = (step * n_batch) % (train_labels.shape[0] - n_batch)
        batch_imgs = train_imgs[offset:(offset + n_batch), :, :, :]
        batch_labels = train_labels[offset:(offset + n_batch), :]
        feed_dict = {tf_train_imgs : batch_imgs, tf_train_labels : batch_labels, drpt:drop_out}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
            steps.append(step)
            print('Minibatch loss at step %d: %f' % (step, l))
            acc = accuracy(predictions, batch_labels)
            batch_acc.append(acc)
            print('Minibatch accuracy: %.1f%%' % acc)
            acc = accuracy(valid_prediction.eval(feed_dict={drpt:1.0}), valid_labels)
            valid_acc.append(acc)
            print('Validation accuracy: %.1f%%' % acc)
    #testdata
    for i in range(0, n_test//1000):
        test_labels[i*1000:(i+1)*1000,:] = test_prediction.eval(
        feed_dict = {tf_test_imgs:test_imgs[i*1000:(i+1)*1000], drpt:1.0})
#%%
plt.plot(steps,batch_acc, label = 'batch_acc')
plt.plot(steps,valid_acc, label = 'valid_acc')
plt.legend(loc='lower right', frameon=False)
plt.ylim(ymax = 100, ymin = 85)
plt.ylabel('accuracy')
plt.xlabel('step')
plt.show()
#%%
test_results = np.argmax(test_labels,1)
print(test_results)
np.savetxt('submission.csv', 
           np.c_[range(1,n_test+1),test_results], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')

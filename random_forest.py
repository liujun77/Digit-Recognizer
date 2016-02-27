# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 13:06:14 2016

@author: liujun
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
#%%

all_tr_data = np.genfromtxt(
'train.csv', dtype=np.int32, delimiter=',', skip_header=True)
result_data = np.genfromtxt(
'test.csv', dtype=np.int32, delimiter=',', skip_header=True)
#%%

image_size = 28
pixel_depth = 255
train_num = int(all_tr_data.shape[0]*0.5)
valid_num = int(all_tr_data.shape[0]*0.3)
test_num = all_tr_data.shape[0] - train_num - valid_num
all_label = all_tr_data[:,0]

train_label = all_label[:train_num]
valid_label = all_label[train_num:train_num+valid_num]
test_label = all_label[train_num+valid_num:]

train_data = all_tr_data[:train_num,1:image_size*image_size+1]
train_data = (train_data.astype(float)-pixel_depth/2.)/pixel_depth
valid_data = all_tr_data[train_num:train_num+valid_num,1:image_size*image_size+1]
valid_data = (valid_data.astype(float)-pixel_depth/2.)/pixel_depth
test_data = all_tr_data[train_num+valid_num:,1:image_size*image_size+1]
test_data = (test_data.astype(float)-pixel_depth/2.)/pixel_depth
result_data = (result_data.astype(float)-pixel_depth/2.)/pixel_depth
#%%
print('train_num =' , train_num, \
'mean =',np.mean(train_data),'stderr =',np.std(train_data) )
print('valid_num =' , valid_num, \
'mean =',np.mean(valid_data),'stderr =',np.std(valid_data) )
print('test_num =' , test_num, \
'mean =',np.mean(test_data),'stderr =',np.std(test_data) )
i=1030
test_image = train_data[i,:]
test_image.shape=(image_size,image_size)
plt.imshow(test_image)
plt.show()
print('label = ',train_label[i])
#%%
stat_label = np.zeros(10)
for i in range(10):
    stat_label[i] = np.sum(test_label==i)
print('stat_label =', stat_label)

#%%
rf = RandomForestClassifier(n_estimators=20, max_features=20)
rf.fit(train_data,train_label)
print('train accuacy =', rf.score(train_data,train_label))
print('valid accuacy =', rf.score(valid_data,valid_label))
#print('test accuacy =', rf.score(test_data,test_label))
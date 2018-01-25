import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import numpy as np
import os

IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic')
train_data = np.load('train_data.npy')

import tensorflow as tf
tf.reset_default_graph()

conv = input_data(shape=[None, IMG_SIZE,IMG_SIZE, 1], name='input')

conv = conv_2d(conv, 32, 5, activation='relu') #nb_filter=32, filter_size=5
conv = max_pool_2d(conv, 5) #kernel_size=5

conv = conv_2d(conv, 64, 5, activation='relu')
conv = max_pool_2d(conv, 5)

conv = conv_2d(conv, 128, 5, activation='relu')
conv = max_pool_2d(conv, 5)

conv = conv_2d(conv, 64, 5, activation='relu')
conv = max_pool_2d(conv, 5)

conv = conv_2d(conv, 32, 5, activation='relu')
conv = max_pool_2d(conv, 5)

conv = fully_connected(conv, 1024, activation='relu')
conv = dropout(conv,0.8)

conv = fully_connected(conv, 2 ,activation='softmax')
cov = regression(conv, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy',name='targets')

model = tflearn.DNN(conv, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')
    
train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

#tensorboard --logdir="./log"
#http://localhost:6006/

#model.save(MODEL_NAME)

import matplotlib.pyplot as plt

test_data = np.load('test_data.npy')

fig=plt.figure()

for num,data in enumerate(test_data[:12]):
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 1: str_label='Dog'
    else: str_label='Cat'
    
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()






import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import glob
import pickle
import numpy as np
import pandas as pd

from src import tfi
import time, sys

# im = cv2.imread('../Train/0/00000_00000_00000.png') 
# print(im.shape)
'''
 function to read and resize images, get labels and store them into np array
def get_image_label_resize(label, filelist, dim = (32, 32), dataset = 'Train'):
    x = np.array([cv2.resize(cv2.imread(fname), dim, interpolation = cv2.INTER_AREA) for fname in filelist])
    y = np.array([label] * len(filelist))
        
    #print('{} examples loaded for label {}'.format(x.shape[0], label))
    return (x, y)    
    
 data for label 0. I store them in parent level so that they won't be uploaded to github
filelist = glob.glob('../Train/'+'0'+'/*.png')
trainx, trainy = get_image_label_resize(0, glob.glob('../Train/'+str(0)+'/*.png'))

 go throgh all others labels and store images into np array
for label in range(1, 43):
    filelist = glob.glob('../Train/'+str(label)+'/*.png')
    x, y = get_image_label_resize(label, filelist)
    trainx = np.concatenate((trainx ,x))
    trainy = np.concatenate((trainy ,y))

 save data into a pickle to later use
trainx.dump('../trainx.npy')
trainy.dump('../trainy.npy')
'''
# load data from pickle
trainx = np.load('../trainx.npy', allow_pickle=True)
trainy = np.load('../trainy.npy', allow_pickle=True)
'''
 get path for test images
testfile = pd.read_csv('Test.csv')['Path'].apply(lambda x: '../' + x).tolist()
 print(testfile)

X_test = np.array([cv2.resize(cv2.imread(fname), (32, 32), interpolation = cv2.INTER_AREA) for fname in testfile])
X_test.dump('../testx.npy')

y_test = np.array(pd.read_csv('Test.csv')['ClassId'])
y_test.dump('../testy.npy')
'''
# load data from pickle
X_test = np.load('../testx.npy', allow_pickle=True)
y_test = np.load('../testy.npy', allow_pickle=True)

# shuffle training data and split them into training and validation
indices = np.random.permutation(trainx.shape[0])
# 20% to val
split_idx = int(trainx.shape[0]*0.8)
train_idx, val_idx = indices[:split_idx], indices[split_idx:]
X_train, X_validation = trainx[train_idx,:], trainx[val_idx,:]
y_train, y_validation = trainy[train_idx], trainy[val_idx]

# get overall stat of the whole dataset
n_train = X_train.shape[0]
n_validation = X_validation.shape[0]
n_test = X_test.shape[0]
image_shape = X_train[0].shape
n_classes = len(np.unique(y_train))
# print("There are {} training examples ".format(n_train))
# print("There are {} validation examples".format(n_validation))
# print("There are {} testing examples".format(n_test))
# print("Image data shape is {}".format(image_shape))
# print("There are {} classes".format(n_classes))

# convert the images to grayscale
X_train_gry = np.sum(X_train/3, axis=3, keepdims=True)
X_validation_gry = np.sum(X_validation/3, axis=3, keepdims=True)
X_test_gry = np.sum(X_test/3, axis=3, keepdims=True)

# Normalize data
X_train_normalized_gry = (X_train_gry-128)/128
X_validation_normalized_gry = (X_validation_gry-128)/128
X_test_normalized_gry = (X_test_gry-128)/128


# update the train, val and test data with normalized gray images
X_train = X_train_normalized_gry
X_validation = X_validation_normalized_gry
X_test = X_test_normalized_gry

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

model = models.Sequential()
# Conv 32x32x1 => 28x28x6.
model.add(layers.Conv2D(filters = 6, kernel_size = (5, 5), strides=(1, 1), padding='valid', 
                        activation='relu', data_format = 'channels_last', input_shape = (32, 32, 1)))
# Maxpool 28x28x6 => 14x14x6
model.add(layers.MaxPooling2D((2, 2)))
# Conv 14x14x6 => 10x10x16
model.add(layers.Conv2D(16, (5, 5), activation='relu'))
# Maxpool 10x10x16 => 5x5x16
model.add(layers.MaxPooling2D((2, 2)))
# Flatten 5x5x16 => 400
model.add(layers.Flatten())
# Fully connected 400 => 120
model.add(layers.Dense(120, activation='relu'))
# Fully connected 120 => 84
model.add(layers.Dense(84, activation='relu'))
# Dropout
model.add(layers.Dropout(0.2))
# Fully connected, output layer 84 => 43
model.add(layers.Dense(43, activation='softmax'))

# specify optimizer, loss function and metric
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training batch_size=128, epochs=10
#conv = model.fit(X_train, y_train, batch_size=128, epochs=10,
#        validation_data=(X_validation, y_validation))

#model.save_weights('h5/traffic-trained.h5')

model.load_weights('h5/traffic-trained.h5')
'''
tesX = []
for i in range(43):
	tesX.append([])

for i in range(len(X_test)):
	loss, acc = model.evaluate(X_test[i:i+1], y_test[i:i+1], verbose=0)
	if(acc == 1.):
		tesX[int(y_test[i:i+1])].append(i)

with open("tesX.txt", "wb") as fp:
	pickle.dump(tesX, fp)
'''

with open("tesX.txt", "rb") as fp:
    tesX = pickle.load(fp)

for i in range(43):
    tesX[i] = tesX[i][:30]

countX = []

for i in range(43):
    countX.append(0.)

start = time.time()

conf = sys.argv[1]
filePath = sys.argv[2]
filePath = os.path.join(filePath, "res.csv")

f = open(filePath, "w")
numFaults = int(sys.argv[3])

for k in range(numFaults):
    for i in range(43):
        count = 0.
        tesXi = tesX[i]
        for j in range(30):
            res = tfi.inject(model=model, x_test=X_test[tesXi[j:j+1]], confFile=conf)
            if (res == i):
                count = count + 1.
        countX[i] = countX[i] + count

for i in range(43):
    countX[i] = countX[i]/numFaults

f.write(str(countX))
f.write("\n")
f.write("Time for %d injections: %f seconds" % (numFaults, time.time() - start))
f.close()

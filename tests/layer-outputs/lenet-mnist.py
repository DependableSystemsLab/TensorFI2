import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import numpy as np

import time, sys, math, random

from src import tensorfi2 as tfi

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images[:, :, :, np.newaxis]
test_images = test_images[:, :, :, np.newaxis]

train_images, test_images = train_images / 255.0, test_images / 255.0
train_images = np.pad(train_images, ((0,0),(2,2),(2,2),(0,0)), 'constant')
test_images = np.pad(test_images, ((0,0),(2,2),(2,2),(0,0)), 'constant')

model = models.Sequential()
model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32,32,1)))
model.add(layers.AveragePooling2D())
model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(layers.AveragePooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(units=120, activation='relu'))
model.add(layers.Dense(units=84, activation='relu'))
model.add(layers.Dense(units=10, activation = 'softmax'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
'''
# Change to True if you want to train from scratch
train = False

if(train):
	# Save the untrained weights for future training with modified dataset
	model.save_weights('h5/lenet-untrained.h5')

	model.fit(train_images, train_labels, batch_size=100, epochs=10,
		validation_data=(test_images, test_labels))

	model.save_weights('h5/lenet-trained.h5')

else:
	model.load_weights('h5/lenet-trained.h5')

	test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
	print("Accuracy before faults:", test_acc)

	tfi.inject(model=model, confFile="confFiles/sample.yaml")

	test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
	print("Accuracy after faults:", test_acc)
'''
conf = sys.argv[1]
filePath = sys.argv[2]
filePath = os.path.join(filePath, "res.csv")

f = open(filePath, "w")
numFaults = int(sys.argv[3])
numInjections = int(sys.argv[4])
offset = 10
num = test_images.shape[0]

totsdc = 0.0

model.load_weights('h5/lenet-trained.h5')
ind = []
init = random.sample(range(num), numInjections+offset)

for i in init:
	test_loss, test_acc = model.evaluate(test_images[i:i+1], test_labels[i:i+1], verbose=0)
	if(test_acc == 1.):
		ind.append(i)
ind = ind[:numInjections]

start = time.time()
for i in range(numFaults):
    model.load_weights('h5/lenet-trained.h5')
    sdc = 0.
    for i in ind:
        res = tfi.inject(model=model, x_test=test_images[i:i+1], confFile=conf)
        if(res != test_labels[i:i+1]):
            sdc = sdc + 1.
    f.write(str(sdc/numInjections))
    f.write("\n")
    totsdc = totsdc + sdc
f.write("\n")
f.write(str(totsdc/(numFaults*numInjections)))
f.write("\n")
f.write("Time for %d injections: %f seconds" % (numFaults*numInjections, time.time() - start))
f.close()

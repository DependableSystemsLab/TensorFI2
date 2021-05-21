import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

import numpy as np
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

import time, sys, math

from src import tensorfi2 as tfi

model = tf.keras.applications.VGG16(
    include_top=True, weights='imagenet', input_tensor=None,
    input_shape=None, pooling=None, classes=1000)

model.compile(optimizer='sgd', loss='categorical_crossentropy')

#model.save_weights('h5/vgg16-trained.h5')

numImages = 10

#https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57
imagesets = ['n02442845', 'n15075141', 'n02457408', 'n03642806', 'n03100240', 'n03792782', 'n03131574', 'n13133613', 'n12144580', 'n02992211']
labels = ['mink', 'toilet_tissue', 'three-toed_sloth', 'laptop', 'convertible', 'mountain_bike', 'crib', 'ear', 'corn', 'cello']
#classes = [23, 889, 38, 228, 268, 255, 298, 329, 331, 342]

images = []
img_labels = []

for i, l in zip(imagesets, labels):
    abspath = '/home/nniranjhana/datasets/imagenet18/validation/'
    abspathi = os.path.join(abspath, i)
    for j in range(numImages):
        rand_file = random.choice(os.listdir(abspathi))
        path = os.path.join(abspathi, rand_file)
        image = load_img(path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        out = model.predict(image)
        label = decode_predictions(out)
        label = label[0][0]
        if(label[1] == l):
            images.append(path)
            img_labels.append(l)

ind = random.sample(range(len(images)), 10)

conf = sys.argv[1]
filePath = sys.argv[2]
filePath = os.path.join(filePath, "res.csv")

f = open(filePath, "w")
numFaults = int(sys.argv[3])
numInjections = 10
#numInjections = int(sys.argv[4])

totsdc = 0.0

start = time.time()
for i in range(numFaults):
    model.load_weights('h5/vgg16-trained.h5')
    sdc = 0.
    for i in ind:
        image = load_img(images[i], target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        res = tfi.inject(model=model, x_test=image, confFile=conf)
        label_ = decode_predictions(res[0])
        label_ = label_[0][0]
        if(label_[1] != img_labels[i]):
            sdc = sdc + 1.
    f.write(str(sdc/numInjections))
    f.write("\n")
    totsdc = totsdc + sdc
f.write("\n")
f.write(str(totsdc/(numFaults*numInjections)))
f.write("\n")
f.write("Time for %d injections: %f seconds" % (numFaults*numInjections, time.time() - start))
f.close()

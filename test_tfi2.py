import os
import sys

import numpy as np
from tensorflow.keras.applications import vgg16, vgg19, resnet, xception, nasnet, mobilenet, mobilenet_v2, \
    inception_resnet_v2, inception_v3, densenet
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from src import tensorfi2 as tfi

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_model_from_name(model_name):
    if model_name == "ResNet50":
        return resnet.ResNet50()
    elif model_name == "ResNet101":
        return resnet.ResNet101()
    elif model_name == "ResNet152":
        return resnet.ResNet152()
    elif model_name == "VGG16":
        return vgg16.VGG16()
    elif model_name == "VGG19":
        return vgg19.VGG19()
    elif model_name == "Xception":
        return xception.Xception()
    elif model_name == "NASNetMobile":
        return nasnet.NASNetMobile()
    elif model_name == "NASNetLarge":
        return nasnet.NASNetLarge()
    elif model_name == "MobileNet":
        return mobilenet.MobileNet()
    elif model_name == "MobileNetV2":
        return mobilenet_v2.MobileNetV2()
    elif model_name == "InceptionResNetV2":
        return inception_resnet_v2.InceptionResNetV2()
    elif model_name == "InceptionV3":
        return inception_v3.InceptionV3()
    elif model_name == "DenseNet121":
        return densenet.DenseNet121()
    elif model_name == "DenseNet169":
        return densenet.DenseNet169()
    elif model_name == "DenseNet201":
        return densenet.DenseNet201()


def get_preprocessed_input_by_model_name(model_name, x_val):
    if model_name == "ResNet50" or model_name == "ResNet101" or model_name == "ResNet152":
        return resnet.preprocess_input(x_val)
    elif model_name == "VGG16":
        return vgg16.preprocess_input(x_val)
    elif model_name == "VGG19":
        return vgg19.preprocess_input(x_val)
    elif model_name == "Xception":
        return xception.preprocess_input(x_val)
    elif model_name == "NASNetMobile" or model_name == "NASNetLarge":
        return nasnet.preprocess_input(x_val)
    elif model_name == "MobileNet":
        return mobilenet.preprocess_input(x_val)
    elif model_name == "MobileNetV2":
        return mobilenet_v2.preprocess_input(x_val)
    elif model_name == "InceptionResNetV2":
        return inception_resnet_v2.preprocess_input(x_val)
    elif model_name == "InceptionV3":
        return inception_v3.preprocess_input(x_val)
    elif model_name == "DenseNet121" or model_name == "DenseNet169" or model_name == "DenseNet201":
        return densenet.preprocess_input(x_val)


def main():
    model_name = sys.argv[1]
    model = get_model_from_name(model_name)
    conf_file = sys.argv[2]
    total_injection = int(sys.argv[3])
    input_dim = int(sys.argv[4])

    # Golder run
    path = 'ILSVRC2012_val_00000001.JPEG'
    image = load_img(path, target_size=(input_dim, input_dim))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = get_preprocessed_input_by_model_name(model_name, image)
    out = model.predict(image).argmax(axis=-1)[0]
    print("Fault free prediction " + str(out))

    # Inject fault to single image
    print("Injecting faults to single image")
    for i in range(total_injection):
        res = tfi.inject(model=model, x_test=image, confFile=conf_file)
        print(res.final_label)

    # Inject fault to batch images
    print("Injecting faults to batch images")
    for i in range(total_injection):
        image = load_img(path, target_size=(input_dim, input_dim))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        batch_image = np.concatenate((image, image), axis=0)
        batch_image = get_preprocessed_input_by_model_name(model_name, batch_image)
        res = tfi.inject(model=model, x_test=batch_image, confFile=conf_file)
        print(res.final_label)
    print("Fault injection done")


if __name__ == '__main__':
    main()

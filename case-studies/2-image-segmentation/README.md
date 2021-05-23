### Case Study 2

#### Visualizing resilience to bit-flips at lower levels of object detection: Is it possible to identify the layer at which bit-flips occur from analysing the faulty masks predicted?

###### 1. Objective and reproducing the study

The goal of this experiment is to visualize how faults in different layers in DNNs affect the outcome of object detection. This is useful for identifying the layer in which the faults occurred. Further, it also provides us with a visual understanding of how the neural network “sees” in the presence of faults.

Image segmentation is an important part of object detection, and has applications in AVs, satellite and medical imaging. We use image segmentation as the target
application in our experiment. We use the modified U-Net from the [TensorFlow tutorial](https://www.tensorflow.org/tutorials/images/segmentation) with the Oxford-IIIT Pet Dataset. The dataset consists of 37 categories of different dog and cat breeds. To identify where an object is located in the image, the image segmentation component outputs a pixel-wise mask of the image. Each pixel of the image is assigned one of three labels, (i) belonging to the pet, (ii) bordering the pet, or (iii) surrounding pixel.

When running [imgseg.py](https://github.com/DependableSystemsLab/TensorFI2/blob/master/case-studies/2-image-segmentation/imgseg.py) for the first time, users have to wait until the Oxford-IIIT Pets dataset is loaded onto disk and model is trained (uncomment relevant lines). Subsequent FI runs will be faster.

###### 2. Experimental details

After we have trained the model, we inject faults into the different layer states of the decoder or upsampler of the UNet. The upsampler encodes the states back into the higher dimensional format using the reverse Conv2DTranspose layers. There are 4 blocks in the upsampler, followed by the last Conv2DTranspose layer which effectively reshapes the image into the original pixel dimensions by convolving over the upsampled data. We inject 100 bit-flips into either one of the 4 Conv2DTranspose layer weights in the upsample blocks, or in the final Conv2DTranspose layer. We will refer to the these Conv2DTranspose layers into which we inject faults as convolutional layers henceforth.

###### 3. Results

The resulting predicted masks for 5 test images in the presence of faults is shown in the figure below (all image results from each configuration can be found in the [fi-img](https://github.com/DependableSystemsLab/TensorFI2/tree/master/case-studies/2-image-segmentation/fi-img) directory). The first and second columns show the original images, and the correct masks for the images respectively. The third to the seventh columns show the faulty masks predicted when faults were injected into one of the five convolutional layers in order.

<h1 align="center"> 
  <img src="https://user-images.githubusercontent.com/29974283/119279149-a6c51600-bbde-11eb-8815-a5320d04844a.png" height="500">
</h1>

We make three main observations. First, we find that faults in the initial layers result in higher disruption (i.e., an unrecognizable mask) compared to faults in the latter layers. This is in line with a previous result from the second case study of TensorFI 1, where we found that faults in initial layers lead to higher SDC rates because they have a longer fault propagation path. This effect is especially pronounced when convolutional operations are involved, where faults in one value propagate to two or more values. The second observation is that there are repeating units of faulty areas in the predicted masks. These are larger for faults in the initial layers, and smaller for faults in the latter layers. To understand this, consider the predicted masks from faults in the first layer. 4x4 dimensional tensors are upsampled to 8x8, 16x16, 32x32, 64x64 and finally 128x128 after each layer. This means that when 100 bit-flips injected into the first layer, they have a higher likelihood of spreading out more by the time they reach the final layer compared to them being injected in the final layer. Since all the five layers have the same filter sizes and strides in our model, we can observe that the dimension of the faulty units gets halved as we move each step to the right to the final layers.

The second observation is that there are repeating units of faulty areas in the predicted masks. These are larger for faults in the initial layers, and smaller for faults in the latter layers. To understand this, consider the predicted masks from faults in the first layer. 4x4 dimensional tensors are upsampled to 8x8, 16x16, 32x32, 64x64 and finally 128x128 after each layer. This means that when 100 bit-flips injected into the first layer, they have a higher likelihood of spreading out more by the time they reach the final layer compared to them being injected in the final layer. Since all the five layers have the same filter sizes and strides in our model, we can observe that the dimension of the faulty units gets halved as we move each step to the right to the final layers.

The third observation is that we can identify the layer at which faults occur based on the final output. Although the faulty masks shown are the outcomes of a single injection, we show different predicted masks for the same fault configuration (of 100 bit-flips) in the first convolutional layer in Figure 20. Each instance produces a different pattern because these are random FIs, and the faults are propagated
differently based on their values. However, across all these different images, the size of the faulty unit is the largest in the first layer. This size depends on the dimension and number of the upsample blocks used in the decoder. We have four blocks and the first upsample block converts a 4x4 to 8x8 image using the Conv2DTranspose operation. In the final faulty outputs in the figure below, we can see the faulty unit repeated 4x4 times. Similarly, for the second upsample block that converts 8x8 images to 16x16, we see the faulty unit repeated 8x8 times in Figure 19 and so on. This observation helps us identify the layer in which the faults originated.

<h1 align="center"> 
  <img src="https://user-images.githubusercontent.com/29974283/119279155-b2b0d800-bbde-11eb-824a-efe9672fadfd.png" height="200">
</h1>
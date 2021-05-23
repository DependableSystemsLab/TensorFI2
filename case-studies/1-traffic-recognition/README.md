### Case Study 1

#### Understanding resilience to bit-flips in a traffic sign recognition model: Are certain classes more vulnerable?

###### 1. Reproducing the study

To run this case study, users have to download the GTSRB dataset. We could not upload datasets because of GitHub file size restrictions. Users have to uncomment the lines that read and store the dataset images into np arrays, train the model and store weights for the first time. Once this is done, for subsequent FI runs, the current [traffic.py](https://github.com/DependableSystemsLab/TensorFI2/tree/master/case-studies/1-traffic-recognition/traffic.py) file can be used as is.

###### 2. Experimental details

We chose dynamic injection of bit-flips in the two convolutional layer outputs. We varied the amount of bit-flips from 10 to 100 in steps of 10. After the model is trained, we chose 30 test inputs from each class of 43 traffic signs in the dataset that were predicted correctly (in the absence of faults). We then performed 1000 FI trials for each test input in each class (results available in [res-conv1](https://github.com/DependableSystemsLab/TensorFI2/tree/master/case-studies/1-traffic-recognition/res-conv1) and [res-conv2](https://github.com/DependableSystemsLab/TensorFI2/tree/master/case-studies/1-traffic-recognition/res-conv2)), and plotted the mean number of images that are predicted correctly (refer [plot-cs1.py](https://github.com/DependableSystemsLab/TensorFI2/tree/master/case-studies/1-traffic-recognition/plot-cs1.py)) in the figure below.

<h1 align="center"> 
  <img src="https://user-images.githubusercontent.com/29974283/119278541-c6f2d600-bbda-11eb-92e9-27953f38ddd9.png" height="400">
</h1>

###### 3. Results

We explain the results from injection in the first convolutional layer. In the figure above, we see that out of 30 test inputs, around 25 get classified correctly for almost all the classes when 10 bit-flips are injected. However, increasing the number of bit-flips decreases the number of images classified correctly (as expected). Further, the difference in the amount of correct predictions grows more pronounced among the different classes as we increase the number of bit-flips.

Next, using the data from the previous figure, we examine the top 5 most and least vulnerable classes of the images in the figure below. We find that **certain classes of images are more vulnerable** than others. Classes 12, 13, 38, 35 and 29 (in that decreasing order) are the most resilient, and have between 9 and 6 images out of 30 classified correctly even in the presence of a 100 bit-flips. Classes 19, 31, 8, 6
and 42 (in that increasing order), are the least resilient, and have between 2 and 4 images out of 30 classified correctly in the presence of 100 bit-flips. Also, there is a significant difference between the percentages of correctly predicted images of the most resilient (Class 12 with 29.03%) and the least resilient (Class 19 with 9.31%) classes.

<h1 align="center"> 
  <img src="https://user-images.githubusercontent.com/29974283/119278552-d40fc500-bbda-11eb-90c8-79e5cef4a6f7.png" height="200">
</h1>

The above figure shows some of the test images belonging to each of these 10 classes, chosen randomly. In general, we find that the more resilient traffic signs are those that are sufficiently unique, and have enough representation in the dataset (refer the relative class frequencies data in the GTSRB dataset). For example, the most resilient Class 12 is the “Priority Road Sign” followed by Class 13, the “Yield Sign”. We find that there is no other yellow diamond sign or inverted triangle sign in the dataset, and that both these classes have enough training images in the dataset. While the mandatory blue road signs (Classes 33 to 40) all have above average resilience, we find the most resilient (Classes 38 and 35) have a high representation in the dataset. An exception is Class 29, which has relatively low occurrence of training images. However, it is only the fifth most resilient class, and its value is only slightly higher than the average.

Though some of the least resilient classes have moderately sufficient representation in the training dataset (Classes 31 and 8), we found that almost all the test images of these two classes as well as Class 19 had minimal brightness, low contrast or were blurry. This could have led to their misclassifications under faults. Classes 6, 31, 41 and 42 are all “No More Restrictions Signs” which are the only crossed out grayscale signs in the dataset. We find that these classes have low resilience values indicating that color is also important for the correct prediction of road signs.
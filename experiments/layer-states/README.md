#### Injection into the layer states

Modifying the layer states is static and is done before the inference runs. This is illustrated in the figure below. The layers `conv_1`, `maxpool_1` and `dense_1` are part of a larger convolutional network. Let us suppose the first convolution layer `conv_1` states are chosen for injection. TensorFI 2 then injects the weights or biases of this layer and stores back the faulty parameters in the model. During inference, the test input passes through the different layer computations, and the fault gets activated when the execution reaches the `conv_1` layer outputs. The fault can then further propagate into the consecutive layer computations and result in a faulty prediction.

<h1 align="center">
<img src="https://user-images.githubusercontent.com/29974283/119272285-0d840880-bbba-11eb-9a9f-d30c89c4b95c.png" height="370">
</h1>

##### Benchmarks and datasets

In this and [tests/layer-outputs](https://github.com/DependableSystemsLab/TensorFI2/blob/master/tests/layer-outputs) directories, we provide the ML models we have tested the tool with. These include smaller models like the simple and convolutional neural networks that work with MNIST and CIFAR-10 datasets to the more complex models like ResNet-50, VGG16, SqueezeNet that use the ImageNet dataset.

In our evaluation, we consider SDC as the standard metric rather than accuracy. This is because we wanted to strictly evaluate the model resilience to the faults alone and so we only considered data points that were predicted correctly without faults for the FI runs. Accuracy does not capture this as it considers overall model performance which includes cases where the model predicts incorrectly because of other reasons that exclude the particular fault under consideration. The code to calculate SDC is embedded in the latter part of these test files.
#### Injection into the layer outputs

Modifying the layer outputs is dynamic and is done during the inference runs. This
is illustrated in the figure below. The layers `conv_1`, `maxpool_1` and `dense_1` are part of a larger convolutional network. Let us suppose the first convolution layer `conv_1` outputs are chosen for injection. TensorFI 2 creates two Keras backend functions `K.func_1` and `K.func_2` that work on the original model without duplication but with the inputs and outputs that we specify. During inference, TensorFI 2 passes the inputs to the `K.func_1` which intercepts the computation at the `conv_1` layer, injects faults into the outputs of the layer computation or the activation states and then passes the outputs into the next `K.func_2`, which feeds them to the immediate next layer, and continues the execution on the rest of the original model. Since `K.func_2` works with the faulty computation, faults can propagate to the model’s output, and result in a faulty prediction.

<h1 align="center">
<img src="https://user-images.githubusercontent.com/29974283/119272916-470a4300-bbbd-11eb-8896-7ca4866b9d18.png" height="370">
</h1>

##### Benchmarks and datasets

In this and [tests/layer-states](https://github.com/DependableSystemsLab/TensorFI2/blob/master/tests/layer-states) directories, we provide the ML models we have tested the tool with. These include smaller models like the simple and convolutional neural networks that work with MNIST and CIFAR-10 datasets to the more complex models like ResNet-50, VGG16, SqueezeNet that use the ImageNet dataset.

In our evaluation, we consider SDC as the standard metric rather than accuracy. This is because we wanted to strictly evaluate the model resilience to the faults alone and so we only considered data points that were predicted correctly without faults for the FI runs. Accuracy does not capture this as it considers overall model performance which includes cases where the model predicts incorrectly because of other reasons that exclude the particular fault under consideration. The code to calculate SDC is embedded in the latter part of these test files.
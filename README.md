# TensorFI 2: A fault injector for TensorFlow 2 applications

TensorFI 2 is a fault injector for TensorFlow 2 applications written in Python 3. We make use of the Keras Model APIs to inject static faults in the layer states and dynamic faults in the layer outputs.
Similar to [TensorFI 1](https://github.com/DependableSystemsLab/TensorFI), TensorFI 2 supports both hardware and software faults with different injection modes.
The fault injector is easily configurable with YAML.


### 1. Dependencies

1. TensorFlow framework (v2.0 or greater)

2. Python (v3 or greater)

3. PyYaml (v3 or greater)

4. Keras framework (part of TensorFlow)

5. numpy package (part of TensorFlow)


### 2. Installation and runs

Following are the installation and usage instructions for a Linux platform.

1. Clone the repository.

    ```
    git clone https://github.com/DependableSystemsLab/TensorFI2.git
    ```

2. Set the python path for the TensorFI 2 project so that it can be executed from other scripts. You can also add it permanently to .bashrc if you prefer.

    ```
    export PYTHONPATH=$PYTHONPATH:$TFI2HOMEPATH
    ```

	where `$TFI2HOMEPATH` might be like `/home/nj/TensorFI2`

3. You can navigate to [conf/](https://github.com/DependableSystemsLab/TensorFI2/tree/master/conf) to check out how to set the fault injection configuration for the tests you plan to run.

4. Let's see an example of how to inject a bit-flip into a layer output in the model. Go to [tests/layer-outputs](https://github.com/DependableSystemsLab/TensorFI2/blob/master/tests/layer-outputs) and set the sample.yaml file in [tests/layer-outputs/confFiles](https://github.com/DependableSystemsLab/TensorFI2/tree/master/tests/layer-outputs/confFiles). If you are running from the examples in this directory, this is the file that gets picked up.

5. Run the test to observe the fault injection. For example, let's say we run the simple neural network example:

    ```
    python nn-mnist.py confFiles/sample.yaml result/ 1 10
    ```

    with the following configuration:

    ```
    Mode: layer_outputs
    Type: bitflips
    Amount: 1
    ```

    This means that a single bit will be flipped in the output tensor of a randomly chosen layer in the model. `result/` is the directory where you want to store the output of the run.
    `1` is the number of fault injections you want to run and `10` is the number of test inputs to evaluate each of the fault injection runs.

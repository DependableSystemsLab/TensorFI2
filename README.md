<h1 align="center"> 
  <img src="https://user-images.githubusercontent.com/29974283/100801172-78f3d700-33dc-11eb-8756-375ddbd740ca.png" height="120">
</h1>

<h2 align="center">TensorFI 2: A fault injector for TensorFlow 2 applications</h2>

[![GitHub license](https://img.shields.io/github/license/DependableSystemsLab/TensorFI2)](https://github.com/DependableSystemsLab/TensorFI2/blob/master/LICENSE)
![GitHub language count](https://img.shields.io/github/languages/count/DependableSystemsLab/TensorFI2)
![GitHub top language](https://img.shields.io/github/languages/top/DependableSystemsLab/TensorFI2)
[![GitHub issues](https://img.shields.io/github/issues/DependableSystemsLab/TensorFI2)](https://github.com/DependableSystemsLab/TensorFI2/issues)

```
By far the greatest danger of Artificial Intelligence is that
people conclude too early that they understand it.
-- Eliezer Yudkowsky
```

Welcome to TensorFI 2's GitHub page!

### Table of Contents

1. [Introduction](#introduction)
2. [Dependencies](#dependencies)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [Contributing](#contributing)

### Introduction

TensorFI 2 is a fault injector for TensorFlow 2 applications written in Python 3. We make use of the Keras Model APIs to inject static faults in the layer states and dynamic faults in the layer outputs.
Similar to [TensorFI 1](https://github.com/DependableSystemsLab/TensorFI), TensorFI 2 supports both hardware and software faults with different injection modes.
The fault injector is easily configurable with YAML.

### Dependencies

1. TensorFlow framework (v2.0 or greater)

2. Python (v3 or greater)

3. PyYaml (v3 or greater)

4. Keras framework (part of TensorFlow)

5. numpy package (part of TensorFlow)


### Architecture

ML models are made up of input data, weight matrices that are learned during training, and activation matrices that are computed from the weights and data. While TensorFI 1 targeted only the activation matrices for fault injection, TensorFI 2 is capable of injecting faults into both the weight and activation matrices. These are the two injection targets respectively.

1. Injection into the *layer states* that hold the learned weights and biases is static and can be done before the inference runs. The methodology is further explained in [experiments/layer-states](https://github.com/DependableSystemsLab/TensorFI2/tree/master/experiments/layer-states) along with the relevant experiments.

2. Injection into the *layer outputs* that hold the activations or computations is dynamic and is done during the inference runs. The methodology is further explained in [experiments/layer-outputs](https://github.com/DependableSystemsLab/TensorFI2/tree/master/experiments/layer-outputs) along with the relevant experiments.

Both types of injection support single and multiple faults along with three types of faults - zeros, random value replacements and bit-flips.

### Installation

Installation has just two steps - downloading the source and adding it to `PATH`.

1. Clone the repository.

    ```
    git clone https://github.com/DependableSystemsLab/TensorFI2.git
    ```

2. Set the python path for the TensorFI 2 project so that it can be executed from other scripts. You can also add it permanently to .bashrc if you prefer.

    ```
    export PYTHONPATH=$PYTHONPATH:$TFI2HOMEPATH
    ```

	where `$TFI2HOMEPATH` might be like `/home/nj/TensorFI2`


### Configuration

Let's see an example of how to inject a bit-flip into a layer state in the model. Go to [experiments/layer-states](https://github.com/DependableSystemsLab/TensorFI2/blob/master/experiments/layer-states) and set the sample.yaml file in [experiments/layer-states/confFiles](https://github.com/DependableSystemsLab/TensorFI2/tree/master/experiments/layer-states/confFiles) with the following configuration:

##### sample.yaml
    
    Target: layer_states
    Mode: single
    Type: bitflips
    Amount: 1
    Bit: N


This means that a single bit (chosen randomly as `N` is specified for `Bit`) will be flipped in the **state** tensor (as `Target` specified is `layer_states`) of **one** randomly chosen layer (as injection `Mode` specified is `single`) in the model.

When running from the examples in this directory, this is the YAML file that gets picked up by the injector.

For further understanding of what each label and values mean, navigate to [conf/](https://github.com/DependableSystemsLab/TensorFI2/tree/master/conf) and check out how to set the fault injection configuration for the tests or experiments you plan to run.

### Usage

Run the test to observe the fault injection. For example, let's say we run the simple neural network example:

    python nn-mnist.py confFiles/sample.yaml result/ 1 10

`result/` is the directory where you want to store the output of the run.
`1` is the number of fault injections you want to run and `10` is the number of test inputs to evaluate each of the fault injection runs.

To use TensorFI 2 in a generic ML model (that is not in [experiments](https://github.com/DependableSystemsLab/TensorFI2/blob/master/experiments/)), just add two lines of code to the model.

1. At the top, import tensorfi2 from [src](https://github.com/DependableSystemsLab/TensorFI2/tree/master/src).

    ```
    from src import tensorfi2 as tfi
    ```

2. After model definition and training, insert the call to TensorFI 2's `inject` function during inference runs.

    ```
    tfi.inject(model=model, confFile=confFiles/sample.yaml)
    ```

`model` is the defined Keras model and `confFile` argument requires the YAML configuration file that contains injection instructions.

The above line of injection code is for injection into the layer states with the particular configuration listed previously. Refer the READMEs in [conf](https://github.com/DependableSystemsLab/TensorFI2/tree/master/conf), [experiments](https://github.com/DependableSystemsLab/TensorFI2/tree/master/experiments) and [case-studies](https://github.com/DependableSystemsLab/TensorFI2/tree/master/case-studies) directories for further usage and experiments with the tool.

### Contributing

We encourage all relevant contributions - in the form of ideas, issues, bug fixes, code and documentation. Thank you for trying out and/or contributing to our tool. Refer the [contributing](https://github.com/DependableSystemsLab/TensorFI2/tree/master/CONTRIBUTING.md) guidelines for more information.
## TensorFI 2 Fault Configurations

We list each of the FI labels that can be configured in the YAML files. The meaning of the labels and their allowed values are also explained.

#### 1. Target

Two kinds of injection are possible with TensorFI 2. These are static and dynamic injections. We specify this using the *Target* label, as it depends on what artifact we inject as well.

1. *layer_states:* The weights and biases of each layer compose of its state and *layer_states* FI is **static** and affect the stored state values of the chosen layer.

2. *layer_outputs:* The final computations or activations of each layer compose of its output and *layer_outputs* FI is **dynamic**. This is analogous to operator output injections (the only kind of injection) in TensorFI 1.

#### 2. Mode

Two kinds of injection modes can be specified in TensorFI 2. These are *single* and *multiple*.

1. *single:*  When this mode is selected, only one layer out of the total n model layers is randomly chosen for injection.

2. *multiple:* Presently, multiple mode means that each and every layer is selected for injection.

#### 3. Type

Three types of faults can be injected with TensorFI 2. These are *zeros*, *random* value replacements and *bitflips*.

1. *zeros:* The chosen tensor value(s) are replaced with zeros.

2. *random:* The chosen tensor value(s) are replaced with random value(s) in the range [0,1).

3. *bitflips:* Bits are flipped in the chosen tensor value(s). When *bitflips* fault type is chosen, the bits to be flipped can be chosen randomly during injection or configured before injection.

#### 4. Amount

The *Amount* of faults are of two types - either the exact number or a percentage.

1. For *bitflips* and *random* value replacement fault types, *Amount* is the exact number of values to be injected with the specified fault. As an example, if 10 is specified as the *Amount* for *bitflips*, then 10 tensor values are randomly chosen of that particular layer under consideration and 1 bit is flipped in each of the 10 values. Thus the range of allowed values are integers between 0 and total number of the tensor values of the layer state or output under consideration.

2. For *zeros*, *Amount* is the percentage of total tensor values and varies from 0% to 100%. As an example, if 10 is specified as the *Amount* for *zeros*, and there are 1000 values in that particular layer state or output tensor under consideration, then 100 tensor values are replaced with zeros.

#### 5. Bit

The *Bit* is specified only if *bitflips* fault type are chosen. If the user wants the bit to be flipped to be chosen randomly by the injector during runtime, *N* is specified. Otherwise, the bit position to be flipped can be indicated with a value between 0 and 31. This is because we assume the models we consider use float32 precision.

### Example configuration files

We provide two example config files and explain the injection for each.

##### sample-1.yaml

```
Target: layer_outputs
Mode: single
Type: bitflips
Amount: 1
Bit: 5
```

This configuration can be used to flip the 5th bit in the output tensor of a randomly chosen layer in the model.

##### sample-2.yaml

```
Target: layer_state
Mode: multiple
Type: zeros
Amount: 10
```

This configuration can be used to replace 10% of each and every layer state tensors of the model with zeros.
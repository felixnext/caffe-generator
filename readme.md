# Caffe Model Generator

This is a generator for Caffe Models. It takes basic yaml files that define the structure of the overall network and concrete prototxt blocks to create a complete prototxt model for caffe (much like less generates css files). The goal is to create complex caffe models, that are still easy to change and to experiment with parameters for training.

> **DISCLAIMER:** This is currently just a side-project and was not tested extensively. I also have some things of the todo list, but cannot guarantee that I will find the time to implement them in the near future. If you have any suggestions or pull requests, please do not hesitate.

This project is currently only tested with **python3**.

## Usage

The usage of the script is pretty straight-forward:

```bash
python3 generator.py MODEL_PATH OUTPUT_PATH [PARAMS]
```

The `MODEL_PATH` defines the path to the yaml file that should be parsed. The `OUTPUT_PATH` defines either the concrete file or the folder where the output should be saved.
The last point allows to pass custom parameters as defined by the loaded yaml file (in order to quickly change the parameters for experiments). For example `--fmaps 128`.

Enthusiasts might also go ahead and install the generator permanently:

```bash
pip install setup.py
caffe-generator MODEL_PATH OUTPUT_PATH [PARAMS]
```

## Format

The format is based on three file types: `yaml`, `proto` and `predef`.

Whereas the first two reference either yaml of prototxt files, the last on references predefined layers that do no need an extra files. This allows to easily build networks without writing prototxt for most of the time (to be fair: input and output probably should be written by hand).
The last one (`predef`) is a reference to various internal blocks that can

### YAML Format

These define the parameters and overall topology of the network.

The yaml structure has the following key words:

#### name

Defines the name of the network or block defined by this yaml.

#### description

General description of this block.

#### params

Defines a key-value list of all parameters that are used in this network. These can be referenced from the blocks and function as single point to change values.

> **NOTE:** These parameters do not support referencing or evaluation expressions

#### blocks

Defines the blocks in the correct order that are added to the network as a yaml list. These blocks can either be other yaml files or prototxt files. It uses the following keywords:

* **name** - the name of the block
* **description** - description of this block (will be shown as comment)
* **file** - the file that is used (without file-type) as path relative to the current yaml file
* **type** - the type of the file. Can either be `yaml` or `proto`
* **prefix** - the prefix added to the prototxt added by this block
* **hide** - defines as `yes` or `no` if this block is hidden (will no be included in topology). Can also be a parameter. Default: `no`.
* **repeat** - the number of times this block should be repeated, The repeat value can also reference parameters (e.g. `::repeat_var`) or even use eval expressions (e.g. `!!::repeat_var * 3`)
* **params** - the parameters for this block. This can be plain values or references to global parameters using `::` (e.g. `fmaps_dense: ::fmaps` references the dense feature maps parameter to the global fmaps parameter). This can also be expressions using `!!` (e.g. `fmaps_bottleneck: "!!::fmaps * 4"`. Note that the quotes are required for yaml to parse correctly). Parameters can even reference the current iteration in the repeat with `::ITER` (note that this value is set 1 no repeat is given).
* **output** - Maps the named output of the network to new names (e.g. `conv-out: block2-out`). Use either number for the input position (zero based) (e.g. `1: conv2-out`) or names for named outputs. This is based on a list (to allow multiple mappings) on two keywords (`in` and `out`). Example, see below. **NOTE:** you might use numbers to set the output at a new index. However if the number currently does not exist, the output will simply append. Use `--debug` flag to check the outputs of your network!

> **NOTE:** All parent parameters are also pushed into the child blocks. However values that are also defined in the chlid/blocks `params` are preferred by the parser.

**Example:**

```yaml
# defines the global network name
name: testnet
description: "Simple Test Network"

# defines global params with default values that are used
params:
  act_fct: PReLU
  fmaps: 32
  test_rep: 5

blocks:
  - name: block1
    file: blocks/stem-layer   # note: might leave prototxt
    type: proto
    prefix: stem  # defines if a prefix should be used for the layers
    repeat: 4     # this will repeat this block for X times (OUTPUT will be the last layer)
    params:
      fmaps: 96
      act_fct: ::act_fct  # reference on the global parameter
    output:
      # maps output at position 0 of this block to the name block1-out
      - in: 0   
        out: block1-out
      # maps the output with name pool-out to the name block1-pool
      - in: pool-out
        out: block1-pool
      # set the output of pool out at position 2 of the output
      - in: pool-out
        out: 2

  - name: block2
    file: blocks/test-block
    type: yaml
    repeat: "!!::test_rep + 1"
    params: # note: this allows to set the global params of the yaml file
      fmaps: "!!::fmaps * 2 + ::ITER"

```

### Prototxt Format

These define the concrete layers as usual with caffe. However they provide some special keywords and notations to allow easy connection of various blocks. The following keywords can be used:

#### INPUT

The `[INPUT]` keywords allow to reference the output of the last block. This can either be referenced by a zero-based index from the previous block in the yaml structure (e.g. `[INPUT:1]`), whereby `[INPUT]` maps to `[INPUT:0]`. Or it can be a named reference, which are preserved over multiple blocks (e.g. `[INPUT:my-layer]`).

#### OUTPUT

The output is defined in brackets as a comma-seperated list at the end of the file. Each entry can either be a direct name of a layer (e.g. `dense/conv3`), a named layer (which is preserved over the blocks for later reference, see INPUT) (e.g. `[my-layer:dense/conv3]`) or the pass-through of an input layer (even though named input layers are always passed through) (e.g. `[INPUT:2]`).

> **NOTE:** If no brackets for the output at the end of the file are provided, the parser will take the last `top` attribute of the last layer in the network.

#### Parameters

Parameters are simply indicated by brackets and reference the parameters defined in the yaml file (e.g. `[fmaps]`). They can also be provided with a default value (e.g. `[fmaps:128]`).

#### Expressions

Expressions are an extension of parameters indicated by double brackets. The inner expression can contain a mathematical expression as well as parameters (e.g. `[[[fmaps] * 10]]`). This allows for example to make the size of feature maps relational to a single parameter.

**Example:**

```prototxt
layer {
  name: "conv"
  type: "Convolution"
  bottom: "[INPUT:0]"
  top: "stem/conv3"
  # for the kernel
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  # for the bias
  param {
    lr_mult: 1.0
    decay_mult: 0.0
  }
  convolution_param {
    # the double points denote the default value if no param is provided
    num_output: [fmaps:128]
    pad: 1
    kernel_size: 3
    stride: 2
    weight_filler {
      type: "xavier"
      std: 0.1
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "conv3/PReLU"
  type: "[act_fct]"
  bottom: "stem/conv3"
  top: "stem/conv3"
}

# defines a special element that can be referenced with OUTPUT in the yaml
# NOTE: if none is given the system will take the name of layer
[conv3/PReLU, [INPUT:1], [dict_name:stem/conv2]]
```

### Predefined Layers

There is a library of predefined layers with certain parameters that are listed here. Each of this is referenced in `yaml` (see examples).

#### Dropout

#### BatchNorm

#### Conv

#### Fully Connected

#### Pooling

#### STEM Block

#### Nvidia Detectnet Input

Defines the typical input layers of the DetectNet Object Detection Pipeline as used in Digits.

**Params:**
* **size_x** - Width of input image
* **size_y** - Height of input images
* **stride** - Overall stride of the network (used to define the grid size - Higher Strides = Larger Grid) [DEFAULT: 16]
* **aug_crop_prob** - Augmentation: [DEFAULT: 0.5]
* **aug_shift_x** - Augmentation: [DEFAULT: 32]
* **aug_shift_y** - Augmentation: [DEFAULT: 32]
* **aug_scale_prob** - Augmentation: [DEFAULT: 0.5]
* **aug_scale_min** - Augmentation: [DEFAULT: 0.8]
* **aug_scale_max** - Augmentation: [DEFAULT: 1.2]
* **aug_flip_prob** - Augmentation: [DEFAULT: 0.5]
* **aug_rotation_prob** - Augmentation: [DEFAULT: 0.4]
* **aug_rotate_dregree** - Augmentation: [DEFAULT: 5.0]
* **aug_hue_rotation_prob** - Augmentation: [DEFAULT: 0.3]
* **aug_hue_rotation** - Augmentation: [DEFAULT: 30.0]
* **aug_desaturation_prob** - Augmentation: [DEFAULT: 0.4]
* **aug_desaturation_max** - Augmentation: [DEFAULT: 0.8]

**Output:**
* **transformed_data** - Transformed/Augmented Image that can be used in the network
* **cvg_block:coverage_block** - The
* **cvg-lbl:coverage-label** - ...

> Note: The outputs are constructed in a way that most user will not need to worry about them, as the named outputs are just used by the detectnet-output. The network will simpy use `[INPUT:0]`.

#### Nvidia DetectNet Output

Defines the typical output layer of the Detectnet object detection pipeline as used in digits.

**Params:**
* **size_x** - Width of input image
* **size_y** - Height of Input images
* **stride** - Overall stride of the network (used to define the grid size - Higher Strides = Larger Grid) [DEFAULT: 16]

## ToDo List

* [ ] Generate an automatic description of the model on various hierarchy levels based on the blocks

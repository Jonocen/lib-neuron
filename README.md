# lib-neuron

`lib-neuron` is a lightweight C neural-network library focused on small projects and educational use.

## Why AI and where is it

!!The Doc is AI made because my english is very Bad, Fixed in future!!

## Modules

- `matrixcalculation`: activations, dense layers, conv2d, maxpool2d
- `layers`: plugin wrappers (`LayerPlugin`) for sequential models
- `models`: sequential training/inference helpers
- `lossfunctions`: MSE/BCE losses and gradients
- `optimizers`: SGD/Adam/RMSProp/Adagrad/AdamW updates

Include everything with:

```c
#include <lib-neuron.h>
```

## Project layout

- `include/` — public headers
- `src/` — implementations

## Docs

- `docs/README.md`
- `docs/Quickstart.md`
- `docs/Examples.md`
- `docs/Training.md`
- `docs/AddOptimizer.md`
- `docs/AddLossFunction.md`
- `docs/AddLayerAndPluginLayer.md`
- `docs/FirstScript.md`
- `docs/APIReference.md`

## Return convention

- `0` means success
- `-1` means invalid input or internal failure

Array data (weights, gradients, activations) is updated in place via pointers.

## Build

Build libraries (static + shared):

```sh
make
```

Build shared library only:

```sh
make shared
```

Clean artifacts:

```sh
make clean
```

## Compile your own program

Use the static archive directly for the simplest setup:

```sh
gcc your_program.c -Iinclude ./libneuron.a -lm -o your_program
```

## Plugin layers + sequential model example

A ready-to-run example was added in:

- `examples/sequential_xor_plugin.c`

It shows how to:

- initialize a `SequentialModel`
- add dense plugin layers with `sequential_model_add_dense`
- choose optimizer and loss at runtime
- run inference with `sequential_model_forward`

Build and run it with:

```sh
make sequential_xor_plugin
./examples/sequential_xor_plugin
```

Minimal usage pattern:

```c
SequentialModel model;
float out[1];
float loss = 0.0f;
OptimizerType optimizer = OPTIMIZER_SGD;
LossFunctionType loss_function = LOSS_BCE;
float learning_rate = 0.05f;

sequential_model_init(&model, 2);
sequential_model_add_dense(&model, 2, 4, ACT_RELU);
sequential_model_add_dense(&model, 4, 1, ACT_SIGMOID);

sequential_model_train_step_with_loss(&model,
									  input,
									  target,
									  out,
									  loss_function,
									  optimizer,
									  learning_rate,
									  NULL,
									  &loss);
sequential_model_forward(&model, input, out);

sequential_model_free(&model);
```

`examples/Other_Exaple.c` is the advanced training example with a deeper network, selectable loss/optimizer, and evaluation metrics.

## Contributing

its a little bit of bad code, if you find a pice of code that is bad please contribute to this project or make a issue report on github. Thanks a lot!!!!

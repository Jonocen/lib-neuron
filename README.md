# lib-neuron

`lib-neuron` is work in progress C-libary for Neuronal Networks. its in a a "Beta" Phase with some vibe code that i will change.

## Status

Work in progress. a LOT of stuff needs to be added and a lot of AI stuff needs help.

## Why AI and where is it

i dont now a bit of the stuff thats needed but may you do, and i wanted to make it acceseble now

AI code is in models and matrixcalculations, so please check

Also added by AI:

!!The Doc is AI made because my english is very Bad, Fixed in future!!

bits auf the layer system, and a bit of overall things i coud not to without.

## Modules

- **matrixcalculation**: layer structure, activations, forward pass, backpropagation
- **lossfunctions**: MSE and BCE losses, plus their gradients
- **optimizers**: SGD and Adam parameter updates
- **models**: simple sequential model helpers

You can include everything with a single header:

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
- `docs/FirstScript.md`
- `docs/APIReference.md`

## Return convention

Most functions return a status code:

- `0` = success
- `-1` = invalid input or internal failure

Array data (weights, gradients, activations) is updated **in place** via pointers.

## Build

The repository includes a Makefile and builds **libraries only**.

Build the static library with:

```sh
make
```

This generates `libneuron.a`.

Build the shared library with:

```sh
make shared
```

This generates `libneuron.so`.

To clean build artifacts:

```sh
make clean
```

## Link example

When compiling your own executable, link against the library and math module:

```sh
gcc your_program.c -Iinclude -L. -lneuron -lm
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
gcc examples/sequential_xor_plugin.c -Iinclude -L. -lneuron -lm -o examples/sequential_xor_plugin
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

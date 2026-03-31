# Examples

Current examples are in `examples/`.

The shipped examples currently focus on dense workflows. Conv2D/MaxPool2D APIs are available and shown below as a minimal snippet.

## Example 3: image input training step

Source: `examples/image_train_example.c`

Build:

```sh
make examples
```

Run:

```sh
./examples/image_train_example 28 28 200 2 0:data/cat1.jpg 0:data/cat2.jpg 1:data/dog1.jpg 1:data/dog2.jpg
```

What it demonstrates:

- tf.data-style dataset calls: normalize, batch, cache, prefetch
- loading and formatting many labeled images into one dataset
- cached contiguous training buffers for model fitting
- training through `image_dataset_train` + `sequential_model_train`

## Build all examples

```sh
make examples
```

## Example 0: simple and compact XOR

Source: `examples/simple_compact.c`

Build:

```sh
make simple_compact
```

Run:

```sh
./examples/simple_compact
```

What it demonstrates:

- minimal `SequentialModel` setup
- one-call training via `sequential_model_train_step`
- selectable optimizer/loss with compact code

## Example 1: classic XOR training

Source: `examples/Other_Exaple.c`

Build:

```sh
make Other_Exaple
```

Run:

```sh
./examples/Other_Exaple
```

What it demonstrates:

- layer-array API (`Layer layers[]`)
- compact training loop with stack-allocated gradients
- loss/optimizer choice through `sequential_train_step`
- low-level layer-array workflow in minimal lines

## Example 2: plugin-based sequential XOR

Source: `examples/sequential_xor_plugin.c`

Build:

```sh
make sequential_xor_plugin
```

Run:

```sh
./examples/sequential_xor_plugin
```

What it demonstrates:

- dynamic `SequentialModel`
- plugin dense layers via `sequential_model_add_dense`
- framework-style flow: `sequential_model_compile` + `sequential_model_train` + `sequential_model_predict`

## Conv2D + MaxPool2D snippet

```c
SequentialModel model;
float output[10];

sequential_model_init(&model, 5);
sequential_model_add_conv2d(&model, 32, 32, 1, 8, 3, 3, 1, 1, ACT_RELU);
sequential_model_add_maxpool2d(&model, 32, 32, 8, 2, 2, 2, 0);
sequential_model_add_conv2d(&model, 16, 16, 8, 16, 3, 3, 1, 1, ACT_RELU);
sequential_model_add_maxpool2d(&model, 16, 16, 16, 2, 2, 2, 0);
sequential_model_add_dense(&model, 8 * 8 * 16, 10, ACT_SIGMOID);

sequential_model_randomize(&model, 0.1f);
sequential_model_forward(&model, image_chw_flat, output);
```

Notes:

- Input/output tensors for conv and pool are flattened CHW.
- You must provide correct dimensions between layers.
- MaxPool2D has no trainable weights, but remains fully compatible with the training pipeline.

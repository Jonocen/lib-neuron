# Examples

Current examples are in `examples/`.

The shipped examples currently focus on dense workflows. Conv2D/MaxPool2D APIs are available and shown below as a minimal snippet.

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

## Example 3: tiny MNIST-style classifier

Source: `examples/mnist_tiny_pgm.c`

Build:

```sh
make mnist_tiny_pgm
```

Run:

```sh
./examples/mnist_tiny_pgm
```

Optional args:

```sh
./examples/mnist_tiny_pgm [epochs] [batch_size] [lr] [samples_per_class]
```

What it demonstrates:

- out-of-box synthetic digit dataset generation via `image_dataset_make_tiny_digits`
- conv/pool backbone with explicit `Flatten` before dense classifier
- training stack equivalent to:
	`Conv(1->4,4x4) -> Pool(5x5) -> Conv(4->4,2x2) -> Pool(2x2) -> Flatten -> Linear(16->32->64->64->10)`
- class decoding with `image_dataset_get_label` and `image_argmax`

Notes:

- No external dataset files are required for this example.
- If you want real image files, use `image_dataset_load_pgm_manifest` / `image_dataset_load_pgm_labeled`.
- The first pooling step uses `5x5` in this C example to satisfy strict pooling geometry checks in the current library.

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

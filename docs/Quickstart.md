# Quickstart

This quickstart shows how to build `lib-neuron`, run shipped examples, and start using the new Conv2D and MaxPool2D APIs.

## 1) Build the library

From project root:

```sh
make
```

This produces `libneuron.a` and `libneuron.so`.

## 2) Build all examples from root

```sh
make examples
```

This builds:

- `examples/simple_compact`
- `examples/sequential_xor_plugin`
- `examples/Other_Exaple`
- `examples/mnist_tiny_pgm`

## 3) Run the examples

```sh
./examples/simple_compact
./examples/sequential_xor_plugin
./examples/Other_Exaple
```

MNIST-style tiny example (out-of-box synthetic digits, conv/pool + flatten):

```sh
make mnist_tiny_pgm
./examples/mnist_tiny_pgm
./examples/mnist_tiny_pgm 25 32 0.001 24
```

The example builds a conv/pool backbone plus dense head and runs without any external dataset files.

`Other_Exaple` is a compact layer-array training example.

## 4) Conv2D + MaxPool2D quick usage

`SequentialModel` supports plugin helpers for dense, conv2d, and maxpool2d layers.

```c
SequentialModel model;

sequential_model_init(&model, 4);
sequential_model_add_conv2d(&model, 28, 28, 1, 8, 3, 3, 1, 1, ACT_RELU);
sequential_model_add_maxpool2d(&model, 28, 28, 8, 2, 2, 2, 0);
/* output shape after pool: 14x14x8 -> 1568 */
sequential_model_add_dense(&model, 14 * 14 * 8, 10, ACT_SIGMOID);
```

Conv/pool buffers are flattened in CHW order.

## 5) Compile your own program

```sh
gcc your_program.c -Iinclude ./libneuron.a -lm -o your_program
```

Include all public APIs with:

```c
#include <lib-neuron.h>
```

For full signatures and shape rules, see `docs/APIReference.md`.

## 6) Train from image files (PGM)

The library includes image helpers in `image_processing.h`.

Typical flow:

1. Build `const char *paths[]` and `int labels[]` for your images.
2. Load them into an `ImageDataset` with one-hot labels.
3. Compile your model.
4. Train with either generic or image-specific helper.

```c
#include <lib-neuron.h>

const char *paths[] = {
	"data/0/img_0001.pgm",
	"data/1/img_0002.pgm",
	"data/2/img_0003.pgm",
	/* ... many more ... */
};
int labels[] = {0, 1, 2 /* ... */};
int num_samples = (int)(sizeof(paths) / sizeof(paths[0]));

ImageDataset ds = {0};
image_dataset_load_pgm_labeled(paths,
							   labels,
							   num_samples,
							   10,      /* num classes */
							   28,
							   28,
							   &ds);

sequential_model_compile(&model,
						 LOSS_BCE,
						 OPTIMIZER_ADAM,
						 0.001f,
						 0.9f,
						 0.999f);

sequential_model_train_image_dataset(&model, &ds, 10, 16, NULL);

image_dataset_free(&ds);
```

Notes:

- All images in one dataset call must have the same `expected_width`/`expected_height`.
- `image_load_pgm` supports grayscale PGM (`P2`, `P5`) and normalizes to `[0, 1]`.

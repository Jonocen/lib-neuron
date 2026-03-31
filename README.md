# lib-neuron

`lib-neuron` is a lightweight C neural-network library focused on small projects and educational use.

## Why AI and where is it

i dont now a bit of the stuff thats needed but may you do, and i wanted to make it acceseble now

AI code is in models and matrixcalculations, so please check

Also added by AI:

!!The Doc is AI made because my english is very Bad, Fixed in future!!

bits auf the layer system, and a bit of overall things i coud not to without.

## Modules

- `matrixcalculation`: activations, dense layers, conv2d, maxpool2d
- `computervison`: jcv-like image loading/resize + CHW float conversion
- `libvideo`: terminal ASCII image/video preview helpers
- `dataset`: tf.data-like image dataset pipeline helpers
- `layers`: plugin wrappers (`LayerPlugin`) for sequential models
- `models`: sequential training/inference helpers
- `lossfunctions`: MSE/BCE losses and gradients
- `optimizers`: SGD/Adam/RMSProp updates

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
- `docs/FirstScript.md`
- `docs/APIReference.md`

## Return convention

- `0` means success
- `-1` means invalid input or internal failure

Array data (weights, gradients, activations) is updated in place via pointers.

## Image input (jcv-like)

The `computervison` module provides a simple jcv-style flow for PPM/PGM images:

- `jcv_imread` modes: unchanged / grayscale / color
- `jcv_resize`: nearest or bilinear
- `jcv_load_image_for_model`: load + optional resize + CHW float tensor

Example:

```c
#include <stdlib.h>
#include <lib-neuron.h>

float *input = NULL;
int input_size = 0;
int channels = 0;

if (jcv_load_image_for_model("sample.ppm",
							 JCV_IMREAD_COLOR,
							 28,
							 28,
							 JCV_INTER_LINEAR,
							 1,
							 &input,
							 &input_size,
							 &channels) == 0) {
	float out[10];
	sequential_model_predict(&model, input, out);
	free(input);
}
```

## Dataset pipeline (tf.data style)

Use `dataset` helpers to configure image pipelines with a familiar flow:

```c
ImageDataset ds;
image_dataset_init(&ds, 28, 28, 1, 10, JCV_IMREAD_GRAYSCALE, JCV_INTER_LINEAR);
/* image_dataset_add(&ds, "path/to/img.jpg", label); repeated for all samples */
image_dataset_map_normalize(&ds);
image_dataset_batch(&ds, 128);
image_dataset_cache(&ds);
image_dataset_prefetch(&ds, 2);
image_dataset_build_cache(&ds);
```

## Show pictures in terminal

`libvideo` can render loaded images as ASCII output directly in terminal:

```c
libvideo_show_image_file_ascii("sample.ppm", JCV_IMREAD_COLOR, 80);
```

You can also play multiple images like a frame sequence:

```c
const char *frames[] = {"f1.pgm", "f2.pgm", "f3.pgm"};
libvideo_play_image_sequence_ascii(frames, 3, JCV_IMREAD_GRAYSCALE, 80, 8, 2);
```

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

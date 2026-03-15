# Training

This page explains the training flow in `lib-neuron`.

## Core pieces

- `matrixcalculation`: `layer_forward`, `layer_backward`, `conv2d_layer_forward`, `maxpool2d_layer_forward`
- `lossfunctions`: `loss_mse`, `loss_mse_grad`, `loss_bce`, `loss_bce_grad`
- `optimizers`: `sgd_optimizer`, `adam_optimizer`, `rmsprop_optimizer`, `adagrad_optimizer`, `adamw_optimizer`
- `models`: sequential helpers

Loss selection in models (`loss_function` parameter accepts `LOSS_MSE`, `LOSS_BCE`, or `LOSS_HUBER`):

- `sequential_model_train_step` / `sequential_train_step`
- `sequential_model_optimize_from_prediction` / `sequential_optimize_from_prediction`

Easy sequential workflow:

- `sequential_model_adam_state_init`
- `sequential_train_config_init_adam` or `sequential_train_config_init_sgd`
- `sequential_model_train_step_cfg`

Framework-style workflow:

- `sequential_model_compile`
- `sequential_model_train`
- `sequential_model_predict`

Conv/pool model building helpers:

- `sequential_model_add_conv2d`
- `sequential_model_add_maxpool2d`
- `sequential_model_add_flatten`
- `sequential_model_set_input_shape2d`
- `sequential_model_add_conv2d_simple`
- `sequential_model_add_maxpool2d_simple`

## Built-in sequential training

Use `sequential_train_step` (array-of-layers API) or `sequential_model_train_step` (plugin API).
Both accept a `loss_function` parameter (`LOSS_MSE`, `LOSS_BCE`, or `LOSS_HUBER`) and full optimizer control.

One-call helper pattern:

```c
sequential_model_train_step(&model,
							input,
							target,
							output,
							LOSS_MSE,
							OPTIMIZER_SGD,
							0.05f,
							NULL,
							&loss);
```

Config-driven pattern (shorter for repeated training):

```c
SequentialTrainConfig cfg;
AdamOptimizerState adam = {0};

sequential_model_adam_state_init(&model, &adam, 0.9f, 0.999f);
sequential_train_config_init_adam(&cfg, LOSS_BCE, 0.005f, &adam);

sequential_model_train_step_cfg(&model, input, target, output, &cfg, &loss);
```

Compile/train/predict pattern:

```c
sequential_model_compile(&model, LOSS_MSE, OPTIMIZER_SGD, 0.05f, 0.9f, 0.999f);
// batch_size=1: stochastic (one update per sample)
sequential_model_train(&model, &x[0][0], &y[0][0], 4, 2, 1, 5000, 1, &loss);
sequential_model_predict(&model, input, output);
```

Compile/train/predict with progress output (print every N%):

```c
sequential_model_compile(&model, LOSS_MSE, OPTIMIZER_SGD, 0.05f, 0.9f, 0.999f);
sequential_model_train_with_progress(&model,
									 &x[0][0],
									 &y[0][0],
									 4,
									 2,
									 1,
									 5000,
									 1,
									 10,   // print every 10%
									 &loss);
```

Mini-batch (batch_size > 1 groups samples before each gradient update):

```c
sequential_model_compile(&model, LOSS_MSE, OPTIMIZER_ADAM, 0.005f, 0.9f, 0.999f);
sequential_model_train(&model, &x[0][0], &y[0][0], 4, 2, 1, 10000, 4, &loss);
```

If you prefer explicit control, you can split it into two calls:

- `sequential_forward` / `sequential_model_forward`
- `sequential_optimize_from_prediction` / `sequential_model_optimize_from_prediction`

Split pattern example:

```c
sequential_model_forward(&model, input, output);
loss = loss_mse(output, target, output_size);
sequential_model_optimize_from_prediction(&model,
										  output,
										  target,
										  LOSS_MSE,
										  OPTIMIZER_ADAM,
										  learning_rate,
										  &adam,
										  NULL);
```

Split pattern with explicit BCE:

```c
sequential_model_forward(&model, input, output);

sequential_model_optimize_from_prediction(&model,
										  output,
										  target,
										  LOSS_BCE,
										  OPTIMIZER_ADAM,
										  learning_rate,
										  &adam,
										  &loss);
```

Both do this sequence each step:

1. Forward pass
2. Compute selected loss (MSE, BCE, or Huber)
3. Compute output gradient
4. Backpropagate layer-by-layer
5. Update weights/biases with the selected optimizer

For maxpool layers, there are no trainable parameters. Backward still propagates gradients to the input positions selected by max pooling.

## Using Adam with sequential helpers

Set optimizer to `OPTIMIZER_ADAM` and pass an initialized `AdamOptimizerState`.

Required Adam state per parameter:

- first moment `m`
- second moment `v`
- global step `t` (must start at 1)

Pseudo-usage:

```c
AdamOptimizerState adam = {
	.m_w = adam_m_w,
	.v_w = adam_v_w,
	.m_b = adam_m_b,
	.v_b = adam_v_b,
	.step = 1,
	.beta1 = 0.9f,
	.beta2 = 0.999f,
};

sequential_train_step(layers, num_layers, input, target, output,
					  grads_w, grads_b,
					  LOSS_BCE,
					  OPTIMIZER_ADAM,
					  learning_rate,
					  &adam,
					  &loss);
```

For stable training, keep `m`, `v`, and `t` persistent across all epochs/batches.

## Using AdamW with sequential helpers

Set optimizer to `OPTIMIZER_ADAMW` and pass an initialized `AdamOptimizerState`.

For AdamW in this API, state usage is the same as Adam:

- `m_w` / `m_b`: first moments
- `v_w` / `v_b`: second moments
- `step`: bias-correction step counter (starts at `1`)
- `beta1`, `beta2`: Adam moments

Pseudo-usage:

```c
AdamOptimizerState adamw = {0};

sequential_model_optimizer_state_init(&model,
									  &adamw,
									  OPTIMIZER_ADAMW,
									  0.9f,
									  0.999f);

sequential_model_train_step(&model,
							input,
							target,
							output,
							LOSS_BCE,
							OPTIMIZER_ADAMW,
							0.001f,
							&adamw,
							&loss);
```

## Using RMSProp with sequential helpers

Set optimizer to `OPTIMIZER_RMSPROP` and pass an initialized `AdamOptimizerState`.

For RMSProp in this API, the state is reused like this:

- `m_w` / `m_b`: RMSProp caches for weights/biases
- `beta1`: RMSProp decay (usually `0.9f`)
- `v_w` / `v_b` and `step`: ignored for RMSProp updates

Pseudo-usage:

```c
AdamOptimizerState rms = {0};

sequential_model_adam_state_init(&model, &rms, 0.9f, 0.999f);

sequential_model_train_step(&model,
							input,
							target,
							output,
							LOSS_BCE,
							OPTIMIZER_RMSPROP,
							0.005f,
							&rms,
							&loss);
```

Keep RMSProp caches persistent across all training steps.

## Practical tips

- Keep initialization small (for XOR, small random weights help convergence).
- Use different learning rates per optimizer.
- For XOR examples: `SGD` works well near `0.05f`, while `Adam` and `RMSProp` are often stable near `0.005f`.
- `Adagrad` is often stable around `0.01f` to `0.05f` for small dense models.
- If training stalls near 0.5 predictions, test different seeds/init scale.
- BCE can work better than MSE for sigmoid-based binary outputs.

## Conv2D/MaxPool2D shape tips

- All conv and pool tensors are flattened CHW (`channel`, `height`, `width`).
- Conv2D output shape:
	- `out_w = ((in_w + 2 * padding - kernel_w) / stride) + 1`
	- `out_h = ((in_h + 2 * padding - kernel_h) / stride) + 1`
- MaxPool2D output shape:
	- `out_w = ((in_w + 2 * padding - pool_w) / stride) + 1`
	- `out_h = ((in_h + 2 * padding - pool_h) / stride) + 1`
- Dimension checks are strict. If `(in + 2*padding - kernel_or_pool)` is not divisible by `stride`, layer creation returns `-1`.
- When connecting to dense layers, flatten to `out_w * out_h * out_channels` (for conv) or `out_w * out_h * channels` (for pool).

Short-form API example (stride as last arg):

```c
SequentialModel model;
sequential_model_init(&model, 8);

sequential_model_set_input_shape2d(&model, 28, 28, 1);
sequential_model_add_conv2d_simple(&model, 1, 16, 3, 2); /* stride=2 */
sequential_model_add_maxpool2d_simple(&model, 2, 2);
```

MNIST tiny example stack used in `examples/mnist_tiny_pgm.c`:

```c
sequential_model_add_conv2d(&model, 28, 28, 1, 4, 4, 4, 1, 0, ACT_LINEAR);
sequential_model_add_maxpool2d(&model, 25, 25, 4, 5, 5, 5, 0);
sequential_model_add_conv2d(&model, 5, 5, 4, 4, 2, 2, 1, 0, ACT_LINEAR);
sequential_model_add_maxpool2d(&model, 4, 4, 4, 2, 2, 2, 0);
sequential_model_add_flatten(&model);
sequential_model_add_dense(&model, 16, 32, ACT_RELU);
sequential_model_add_dense(&model, 32, 64, ACT_RELU);
sequential_model_add_dense(&model, 64, 64, ACT_RELU);
sequential_model_add_dense(&model, 64, 10, ACT_LINEAR);
```

## Image dataset training (PGM)

Use `image_processing.h` when training from image files.

Main helpers:

- `image_load_pgm`: load one grayscale PGM (`P2`/`P5`) as normalized float pixels.
- `image_dataset_load_pgm_labeled`: load many image paths with integer labels into contiguous train arrays.
- `image_dataset_load_pgm_manifest`: load dataset directly from a manifest text file.
- `sequential_model_train_image_dataset`: train directly from `ImageDataset`.
- `sequential_model_predict_pgm`: load one image + predict in one call.

### Load more than 10 images into arrays

```c
const char *paths[] = {
	"train/0/a0.pgm", "train/0/a1.pgm", "train/0/a2.pgm",
	"train/1/b0.pgm", "train/1/b1.pgm", "train/1/b2.pgm",
	"train/2/c0.pgm", "train/2/c1.pgm", "train/2/c2.pgm",
	"train/3/d0.pgm", "train/3/d1.pgm", "train/3/d2.pgm"
};

int labels[] = {
	0, 0, 0,
	1, 1, 1,
	2, 2, 2,
	3, 3, 3
};

int num_samples = (int)(sizeof(paths) / sizeof(paths[0]));
```

You can scale this pattern to hundreds or thousands of entries.

### Build dataset and train

```c
ImageDataset ds = {0};
float final_loss = 0.0f;

if (image_dataset_load_pgm_labeled(paths,
								   labels,
								   num_samples,
								   4,     /* num_classes */
								   28,
								   28,
								   &ds) != 0) {
	/* handle error */
}

sequential_model_compile(&model,
						 LOSS_BCE,
						 OPTIMIZER_ADAM,
						 0.001f,
						 0.9f,
						 0.999f);

/* Wrapper API */
sequential_model_train_image_dataset(&model, &ds, 20, 16, &final_loss);

/* Equivalent generic API */
sequential_model_train(&model,
					   ds.inputs,
					   ds.targets,
					   ds.num_samples,
					   ds.input_size,
					   ds.target_size,
					   20,
					   16,
					   &final_loss);

image_dataset_free(&ds);
```

Manifest-based shortcut:

```c
image_dataset_load_pgm_manifest("data/train_manifest.txt",
								10,
								28,
								28,
								&ds);
```

Each manifest row is:

```text
relative/or/absolute/path.pgm 7
```

### Raw bytes to floats

If you already have `uint8` image bytes in memory, convert them with:

```c
image_convert_u8_to_f32(src_u8, width * height, dst_f32);
```

This normalizes to `[0, 1]` and avoids duplicating conversion loops in user code.

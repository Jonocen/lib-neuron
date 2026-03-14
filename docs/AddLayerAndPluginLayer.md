# Add a New Layer and Plugin Layer (Step by Step)

This guide explains how to add a new trainable or non-trainable layer to `lib-neuron`, and then expose it as a `LayerPlugin` for `SequentialModel`.

Use this checklist in order.

## 1. Add the core layer type

Edit `include/matrixcalculation.h`.

- Add a new layer struct.
- Include geometry, caches, and trainable params.

Example (1D batch norm style skeleton):

```c
typedef struct {
    int channels;
    float epsilon;
    float *gamma;
    float *beta;
    float *cache_input;
    float *cache_norm;
} Norm1DLayer;
```

## 2. Declare core layer API

Still in `include/matrixcalculation.h`, add the four standard functions:

- `*_init`
- `*_free`
- `*_forward`
- `*_backward`

Example:

```c
int norm1d_layer_init(Norm1DLayer *layer, int channels, float epsilon);
void norm1d_layer_free(Norm1DLayer *layer);
int norm1d_layer_forward(Norm1DLayer *layer, const float *input, float *output);
int norm1d_layer_backward(const Norm1DLayer *layer,
                          const float *delta_in,
                          float *delta_out,
                          float *grad_w,
                          float *grad_b);
```

## 3. Implement core math

Edit `src/matrixcalculation.c`.

Rules:

- Validate all pointers and dimensions.
- Return `0` on success and `-1` on invalid/failure.
- Use caches from forward in backward.
- Keep gradient sizes aligned with plugin expectation.

Gradient conventions in this project:

- `grad_w` holds parameter gradients for weights-like parameters.
- `grad_b` holds parameter gradients for biases-like parameters.
- For non-trainable layers, use dummy params and set gradients predictably.

## 4. Add plugin constructor declaration

Edit `include/layers.h`.

Add a constructor declaration:

```c
int layer_plugin_norm1d_create(int channels,
                               float epsilon,
                               LayerPlugin *out_plugin);
```

## 5. Implement plugin adapter in `src/layers.c`

Add static adapter functions for your layer:

- `*_forward`
- `*_backward`
- `*_input_size`
- `*_output_size`
- `*_weights`
- `*_biases`
- `*_weights_size`
- `*_biases_size`
- `*_destroy`

Then implement `layer_plugin_*_create(...)`.

Pattern:

1. Allocate context layer (`calloc`/`malloc`).
2. Call `*_init`.
3. Fill every `LayerPlugin` function pointer.
4. Return `0`.
5. On any failure, free and return `-1`.

## 6. Trainable vs non-trainable mapping

### Trainable layer

- `weights()` / `biases()` return real parameter buffers.
- `weights_size()` / `biases_size()` return actual counts.

### Non-trainable layer

- Provide dummy 1-element buffers to satisfy optimizer pipeline.
- Return size `1` for both weights and biases.
- `backward` should set `grad_w[0]` and `grad_b[0]` to `0.0f`.

This matches current maxpool behavior.

## 7. Add SequentialModel helper (optional but recommended)

If you want ergonomic API like dense/conv/pool helpers, edit:

- `include/models.h`
- `src/models.c`

Add a wrapper similar to `sequential_model_add_conv2d`:

```c
int sequential_model_add_norm1d(SequentialModel *model, int channels, float epsilon);
```

Implementation steps:

1. Create `LayerPlugin layer = {0};`
2. Call `layer_plugin_norm1d_create(...)`
3. Call `sequential_model_add_layer(model, layer)`
4. On failure, call `layer_plugin_free(&layer)`

## 8. Check shape and memory contracts

Before merging, verify:

1. `input_size()` and `output_size()` are correct.
2. Forward/backward respect flattened layout if using CHW.
3. `*_free` cleans all allocations.
4. `destroy` in plugin frees context exactly once.
5. No mismatch between `weights_size` and gradient length used in backward.

## 9. Update docs

Minimum files:

- `docs/APIReference.md`
- `docs/Training.md` (if behavior affects training usage)
- this file (`docs/AddLayerAndPluginLayer.md`) when process changes

## 10. Sanity tests

Run this checklist:

1. `make`
2. Forward pass on a tiny input.
3. One training step with `sequential_model_train_step`.
4. Confirm no crashes in `sequential_model_free`.
5. Confirm gradient buffers are finite and expected size.

## 11. Minimal plugin create skeleton

```c
int layer_plugin_norm1d_create(int channels,
                               float epsilon,
                               LayerPlugin *out_plugin) {
    if (!out_plugin) return -1;

    Norm1DLayer *layer = calloc(1, sizeof(Norm1DLayer));
    if (!layer) return -1;

    if (norm1d_layer_init(layer, channels, epsilon) != 0) {
        free(layer);
        return -1;
    }

    out_plugin->ctx = layer;
    out_plugin->forward = norm1d_forward;
    out_plugin->backward = norm1d_backward;
    out_plugin->input_size = norm1d_input_size;
    out_plugin->output_size = norm1d_output_size;
    out_plugin->weights = norm1d_weights;
    out_plugin->biases = norm1d_biases;
    out_plugin->weights_size = norm1d_weights_size;
    out_plugin->biases_size = norm1d_biases_size;
    out_plugin->destroy = norm1d_destroy;

    return 0;
}
```

# API Reference

This page documents every public function in `lib-neuron` and explains the role of each parameter.

## Return codes

Most functions use this convention:

- `0` means success
- `-1` means invalid input and/or internal failure

Loss functions return `float` values directly.

## Naming conventions

Model training helpers use these patterns:

- `*_with_loss`: runtime-selectable loss (`LOSS_MSE` or `LOSS_BCE`)
- `*_mse` / `*_bce`: explicit named wrappers
- no suffix (for example `sequential_train_step`): compatibility helper defaulting to MSE

## Common types

### `Activation`

Defined in `include/matrixcalculation.h`.

- `ACT_LINEAR`: no nonlinearity
- `ACT_RELU`: `max(0, x)`
- `ACT_SIGMOID`: `1 / (1 + exp(-x))`
- `ACT_TANH`: hyperbolic tangent

### `Layer`

Dense layer with cached values used for backprop.

Fields:

- `input_size`: number of inputs
- `output_size`: number of neurons
- `activation`: neuron activation type
- `weights`: flattened `[output_size * input_size]` matrix
- `biases`: `[output_size]`
- `cache_input`: last forward input, used by backward
- `cache_z`: last pre-activation values, used by backward

### `LayerPlugin`

Plugin interface used by `SequentialModel`.

- `ctx`: plugin-owned context object
- `forward`, `backward`: function pointers for pass operations
- `input_size`, `output_size`: dynamic shape getters
- `weights`, `biases`: parameter pointer getters
- `weights_size`, `biases_size`: parameter length getters
- `destroy`: cleanup function for `ctx`

### `OptimizerType`

Defined in `include/optimizers.h`.

- `OPTIMIZER_SGD`
- `OPTIMIZER_ADAM`

### `LossFunctionType`

Defined in `include/lossfunctions.h`.

- `LOSS_MSE`
- `LOSS_BCE`

### `SequentialModel`

Dynamic container of plugin layers.

- `layers`: dynamic array of `LayerPlugin`
- `num_layers`: current number of layers
- `capacity`: allocated plugin slots

### `AdamOptimizerState`

Persistent state for Adam used by sequential helpers.

- `m_w`: first moment vectors for each layer weights
- `v_w`: second moment vectors for each layer weights
- `m_b`: first moment vectors for each layer biases
- `v_b`: second moment vectors for each layer biases
- `step`: global optimization step, starts at `1` and increments each update
- `beta1`: first-moment decay, usually `0.9f`
- `beta2`: second-moment decay, usually `0.999f`

### `SequentialTrainConfig`

Compact training config for sequential model helpers.

- `loss_function`: `LOSS_MSE` or `LOSS_BCE`
- `optimizer`: `OPTIMIZER_SGD` or `OPTIMIZER_ADAM`
- `learning_rate`: step size
- `adam_state`: `NULL` for SGD, required for Adam

## `matrixcalculation.h`

### `float act_apply(float x, Activation a)`

Applies activation function `a` to scalar `x`.

Parameters:

- `x`: input scalar
- `a`: activation type to apply

Returns:

- activated scalar value

### `float act_deriv(float x, Activation a)`

Computes derivative of activation `a` at scalar `x`.

Parameters:

- `x`: input scalar (for layer code this is usually pre-activation `z`)
- `a`: activation type

Returns:

- derivative value at `x`

### `int layer_init(Layer *layer, int input_size, int output_size, Activation activation)`

Allocates and initializes a dense layer and all internal buffers.

Parameters:

- `layer`: output layer object to initialize
- `input_size`: number of input features
- `output_size`: number of output neurons
- `activation`: activation used by this layer

Returns:

- `0` on success, `-1` on invalid input/allocation failure

### `void layer_free(Layer *layer)`

Frees memory owned by a layer.

Parameters:

- `layer`: layer to clean up (safe to call with partially initialized layer)

### `int layer_forward(Layer *layer, const float *input, float *output)`

Forward pass for one dense layer.

Parameters:

- `layer`: initialized layer instance
- `input`: pointer to `[input_size]`
- `output`: pointer to `[output_size]` where activations are written

Returns:

- `0` on success, `-1` on invalid input

### `int layer_backward(const Layer *layer, const float *delta_in, float *delta_out, float *grad_w, float *grad_b)`

Backward pass for one layer.

Parameters:

- `layer`: layer with valid forward caches (`cache_input`, `cache_z`)
- `delta_in`: upstream gradient for this layer outputs, size `[output_size]`
- `delta_out`: optional downstream gradient for previous layer inputs, size `[input_size]` or `NULL` for first layer
- `grad_w`: output buffer for weight gradients, size `[output_size * input_size]`
- `grad_b`: output buffer for bias gradients, size `[output_size]`

Returns:

- `0` on success, `-1` on invalid input/allocation failure

## `layers.h`

### `int layer_plugin_dense_create(int input_size, int output_size, Activation activation, LayerPlugin *out_plugin)`

Creates a dense plugin layer backed by `Layer`.

Parameters:

- `input_size`: input dimension
- `output_size`: output dimension
- `activation`: activation function
- `out_plugin`: output plugin object receiving function pointers and context

Returns:

- `0` on success, `-1` on failure

### `void layer_plugin_free(LayerPlugin *plugin)`

Frees plugin context and resets function pointers.

Parameters:

- `plugin`: plugin to release

## `lossfunctions.h`

### `float loss_mse(const float *pred, const float *target, int size)`

Mean Squared Error over `size` elements.

Parameters:

- `pred`: prediction vector
- `target`: expected vector
- `size`: number of elements

Returns:

- average MSE value

### `int loss_mse_grad(const float *pred, const float *target, int size, float *grad_out)`

Gradient of MSE with respect to predictions.

Parameters:

- `pred`: prediction vector
- `target`: expected vector
- `size`: number of elements
- `grad_out`: output gradient buffer of size `[size]`

Returns:

- `0` on success, `-1` on invalid input

### `float loss_bce(const float *pred, const float *target, int size)`

Binary Cross-Entropy over `size` elements.

Parameters:

- `pred`: prediction probabilities in `[0, 1]`
- `target`: target labels/probabilities
- `size`: number of elements

Returns:

- average BCE value

### `int loss_bce_grad(const float *pred, const float *target, int size, float *grad_out)`

Gradient of BCE with respect to predictions.

Parameters:

- `pred`: prediction probabilities in `[0, 1]`
- `target`: target labels/probabilities
- `size`: number of elements
- `grad_out`: output gradient buffer of size `[size]`

Returns:

- `0` on success, `-1` on invalid input

## `optimizers.h`

### `int adam_optimizer(float *weights, float *grads, float *m, float *v, float beta1, float beta2, float learning_rate, int t, int size)`

Applies one Adam update to parameter vector.

Parameters:

- `weights`: parameter vector updated in place, size `[size]`
- `grads`: gradient vector for current step, size `[size]`
- `m`: first-moment buffer (persistent), size `[size]`
- `v`: second-moment buffer (persistent), size `[size]`
- `beta1`: first-moment decay, typically `0.9f`
- `beta2`: second-moment decay, typically `0.999f`
- `learning_rate`: optimizer step size
- `t`: 1-based optimization step for bias correction
- `size`: number of parameters

Returns:

- `0` on success, `-1` on invalid input

### `int sgd_optimizer(float *weights, float *grads, float learning_rate, int size)`

Applies one SGD update to parameter vector.

Parameters:

- `weights`: parameter vector updated in place, size `[size]`
- `grads`: gradient vector for current step, size `[size]`
- `learning_rate`: optimizer step size
- `size`: number of parameters

Returns:

- `0` on success, `-1` on invalid input

## `models.h`

### `int sequential_model_init(SequentialModel *model, int initial_capacity)`

Initializes an empty dynamic sequential model.

Parameters:

- `model`: model object to initialize
- `initial_capacity`: initial slot count for plugin layers

Returns:

- `0` on success, `-1` on invalid input/allocation failure

### `void sequential_model_free(SequentialModel *model)`

Frees all layers and internal memory of `SequentialModel`.

Parameters:

- `model`: model to clean up

### `int sequential_model_add_layer(SequentialModel *model, LayerPlugin layer)`

Adds plugin layer to model.

Parameters:

- `model`: destination sequential model
- `layer`: plugin layer to move into model ownership

Returns:

- `0` on success, `-1` on invalid input/allocation failure

### `int sequential_model_add_dense(SequentialModel *model, int input_size, int output_size, Activation activation)`

Convenience API to create and append one dense layer plugin.

Parameters:

- `model`: destination sequential model
- `input_size`: dense layer input dimension
- `output_size`: dense layer output dimension
- `activation`: dense layer activation

Returns:

- `0` on success, `-1` on failure

### `int sequential_model_forward(SequentialModel *model, const float *input, float *output)`

Runs forward pass across all plugin layers in model.

Parameters:

- `model`: model containing at least one layer
- `input`: input vector for first layer
- `output`: output vector for last layer (caller-provided buffer)

Returns:

- `0` on success, `-1` on invalid input/failure

### `int sequential_model_predict(SequentialModel *model, const float *input, float *output)`

Alias for `sequential_model_forward`.

### `int sequential_model_compile(SequentialModel *model, LossFunctionType loss_function, OptimizerType optimizer, float learning_rate, float adam_beta1, float adam_beta2)`

Stores training settings inside model and allocates Adam state internally when needed.

### `int sequential_model_train(SequentialModel *model, const float *inputs, const float *targets, int num_samples, int input_size, int target_size, int epochs, float *final_loss_out)`

Runs repeated training using settings from `sequential_model_compile`.

### `int sequential_model_fit(...)`

Backward-compatible alias for `sequential_model_train`.

### `void sequential_train_config_init_sgd(SequentialTrainConfig *cfg, LossFunctionType loss_function, float learning_rate)`

Initializes a config for SGD training.

### `void sequential_train_config_init_adam(SequentialTrainConfig *cfg, LossFunctionType loss_function, float learning_rate, AdamOptimizerState *adam_state)`

Initializes a config for Adam training.

### `int sequential_model_adam_state_init(SequentialModel *model, AdamOptimizerState *out_state, float beta1, float beta2)`

Allocates Adam state buffers based on model layer sizes.

### `void sequential_model_adam_state_free(SequentialModel *model, AdamOptimizerState *state)`

Frees Adam state buffers created by `sequential_model_adam_state_init`.

### `int sequential_model_train_step_cfg(SequentialModel *model, const float *input, const float *target, float *output, const SequentialTrainConfig *cfg, float *loss_out)`

Compact one-call training helper driven by `SequentialTrainConfig`.

### `int sequential_model_train_step_sgd(SequentialModel *model, const float *input, const float *target, float *output, float learning_rate, float *loss_out)`

Legacy convenience helper for one MSE + SGD step.

Parameters:

- `model`: sequential model
- `input`: input vector
- `target`: expected output vector
- `output`: output buffer receiving current prediction
- `learning_rate`: SGD learning rate
- `loss_out`: optional pointer receiving MSE loss (`NULL` to ignore)

Returns:

- `0` on success, `-1` on invalid input/failure

### `int sequential_model_train_step(SequentialModel *model, const float *input, const float *target, float *output, OptimizerType optimizer, float learning_rate, AdamOptimizerState *adam_state, float *loss_out)`

One-call training helper (forward + backward + update) using MSE and selected optimizer.

Parameters:

- `model`: sequential model
- `input`: input vector
- `target`: expected output vector
- `output`: output buffer for current prediction
- `optimizer`: `OPTIMIZER_SGD` or `OPTIMIZER_ADAM`
- `learning_rate`: learning rate for selected optimizer
- `adam_state`: required for `OPTIMIZER_ADAM`, ignored for `OPTIMIZER_SGD`
- `loss_out`: optional pointer receiving MSE loss (`NULL` to ignore)

Returns:

- `0` on success, `-1` on invalid input/failure

### `int sequential_model_train_step_with_loss(SequentialModel *model, const float *input, const float *target, float *output, LossFunctionType loss_function, OptimizerType optimizer, float learning_rate, AdamOptimizerState *adam_state, float *loss_out)`

One-call helper with selectable loss and optimizer.

Parameters:

- `loss_function`: `LOSS_MSE` or `LOSS_BCE`
- all other parameters match `sequential_model_train_step`

Returns:

- `0` on success, `-1` on invalid input/failure

### `int sequential_model_train_step_mse(...)`

Named wrapper for `sequential_model_train_step_with_loss(..., LOSS_MSE, ...)`.

### `int sequential_model_train_step_bce(...)`

Named wrapper for `sequential_model_train_step_with_loss(..., LOSS_BCE, ...)`.

### `int sequential_model_optimize_from_prediction(SequentialModel *model, const float *prediction, const float *target, OptimizerType optimizer, float learning_rate, AdamOptimizerState *adam_state, float *loss_out)`

Backward + update only, intended after explicit `sequential_model_forward`.

Parameters:

- `model`: sequential model with valid cached activations from latest forward
- `prediction`: latest model output (from corresponding forward pass)
- `target`: expected output vector
- `optimizer`: `OPTIMIZER_SGD` or `OPTIMIZER_ADAM`
- `learning_rate`: learning rate for selected optimizer
- `adam_state`: required for `OPTIMIZER_ADAM`, ignored for `OPTIMIZER_SGD`
- `loss_out`: optional pointer receiving MSE loss (`NULL` to ignore)

Returns:

- `0` on success, `-1` on invalid input/failure

### `int sequential_model_optimize_from_prediction_with_loss(SequentialModel *model, const float *prediction, const float *target, LossFunctionType loss_function, OptimizerType optimizer, float learning_rate, AdamOptimizerState *adam_state, float *loss_out)`

Backward+update with selectable loss after explicit forward pass.

### `int sequential_model_optimize_from_prediction_mse(...)`

Named wrapper for `sequential_model_optimize_from_prediction_with_loss(..., LOSS_MSE, ...)`.

### `int sequential_model_optimize_from_prediction_bce(...)`

Named wrapper for `sequential_model_optimize_from_prediction_with_loss(..., LOSS_BCE, ...)`.

### `int sequential_forward(Layer *layers, int num_layers, const float *input, float *output)`

Forward pass for array-of-`Layer` API.

Parameters:

- `layers`: pointer to first layer in stack
- `num_layers`: number of layers in stack
- `input`: input vector for first layer
- `output`: output vector for last layer

Returns:

- `0` on success, `-1` on invalid input/failure

### `int sequential_train_step_sgd(Layer *layers, int num_layers, const float *input, const float *target, float *output, float **grads_w, float **grads_b, float learning_rate, float *loss_out)`

Legacy one-call training helper using MSE + SGD.

Parameters:

- `layers`: layer stack
- `num_layers`: number of layers
- `input`: input vector
- `target`: expected output vector
- `output`: output buffer for current prediction
- `grads_w`: per-layer weight gradient buffers
- `grads_b`: per-layer bias gradient buffers
- `learning_rate`: SGD learning rate
- `loss_out`: optional pointer receiving MSE loss (`NULL` to ignore)

Returns:

- `0` on success, `-1` on invalid input/failure

### `int sequential_train_step(Layer *layers, int num_layers, const float *input, const float *target, float *output, float **grads_w, float **grads_b, OptimizerType optimizer, float learning_rate, AdamOptimizerState *adam_state, float *loss_out)`

One-call training helper (forward + backward + update) for array-of-`Layer` API.

Parameters:

- `layers`: layer stack
- `num_layers`: number of layers
- `input`: input vector
- `target`: expected output vector
- `output`: output buffer for current prediction
- `grads_w`: per-layer weight gradient buffers (caller-allocated)
- `grads_b`: per-layer bias gradient buffers (caller-allocated)
- `optimizer`: `OPTIMIZER_SGD` or `OPTIMIZER_ADAM`
- `learning_rate`: learning rate for selected optimizer
- `adam_state`: required for `OPTIMIZER_ADAM`, ignored for `OPTIMIZER_SGD`
- `loss_out`: optional pointer receiving MSE loss (`NULL` to ignore)

Returns:

- `0` on success, `-1` on invalid input/failure

### `int sequential_train_step_with_loss(Layer *layers, int num_layers, const float *input, const float *target, float *output, float **grads_w, float **grads_b, LossFunctionType loss_function, OptimizerType optimizer, float learning_rate, AdamOptimizerState *adam_state, float *loss_out)`

One-call helper with selectable loss and optimizer for array-of-`Layer` API.

### `int sequential_train_step_mse(...)`

Named wrapper for `sequential_train_step_with_loss(..., LOSS_MSE, ...)`.

### `int sequential_train_step_bce(...)`

Named wrapper for `sequential_train_step_with_loss(..., LOSS_BCE, ...)`.

### `int sequential_optimize_from_prediction(Layer *layers, int num_layers, const float *prediction, const float *target, float **grads_w, float **grads_b, OptimizerType optimizer, float learning_rate, AdamOptimizerState *adam_state, float *loss_out)`

Backward + update only for array-of-`Layer` API, intended after `sequential_forward`.

Parameters:

- `layers`: layer stack with valid forward caches
- `num_layers`: number of layers
- `prediction`: latest model output from corresponding forward pass
- `target`: expected output vector
- `grads_w`: per-layer weight gradient buffers
- `grads_b`: per-layer bias gradient buffers
- `optimizer`: `OPTIMIZER_SGD` or `OPTIMIZER_ADAM`
- `learning_rate`: learning rate for selected optimizer
- `adam_state`: required for `OPTIMIZER_ADAM`, ignored for `OPTIMIZER_SGD`
- `loss_out`: optional pointer receiving MSE loss (`NULL` to ignore)

Returns:

- `0` on success, `-1` on invalid input/failure

### `int sequential_optimize_from_prediction_with_loss(Layer *layers, int num_layers, const float *prediction, const float *target, float **grads_w, float **grads_b, LossFunctionType loss_function, OptimizerType optimizer, float learning_rate, AdamOptimizerState *adam_state, float *loss_out)`

Backward+update with selectable loss after explicit forward pass.

### `int sequential_optimize_from_prediction_mse(...)`

Named wrapper for `sequential_optimize_from_prediction_with_loss(..., LOSS_MSE, ...)`.

### `int sequential_optimize_from_prediction_bce(...)`

Named wrapper for `sequential_optimize_from_prediction_with_loss(..., LOSS_BCE, ...)`.

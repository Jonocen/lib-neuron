# API Reference

This is the complete public API reference for `lib-neuron`.

Scope:
- `include/matrixcalculation.h`
- `include/layers.h`
- `include/lossfunctions.h`
- `include/optimizers.h`
- `include/models.h`

## Conventions

- Most non-loss APIs return `0` on success and `-1` on invalid input or internal failure.
- Loss value functions (`loss_mse`, `loss_bce`) return a `float`.
- Conv/Pool tensors use flattened CHW layout (`channel`, `height`, `width`).

## Core Types

### `Activation`

- `ACT_LINEAR`: no non-linearity.
- `ACT_RELU`: ReLU activation.
- `ACT_SIGMOID`: sigmoid activation.
- `ACT_TANH`: tanh activation.

### `OptimizerType`

- `OPTIMIZER_SGD`
- `OPTIMIZER_ADAM`
- `OPTIMIZER_RMSPROP`

### `LossFunctionType`

- `LOSS_MSE`
- `LOSS_BCE`

## `matrixcalculation.h`

### Activation functions

`float act_apply(float x, Activation a)`
- Applies activation `a` to scalar `x`.

`float act_deriv(float x, Activation a)`
- Returns derivative of activation `a` at scalar `x`.

### Dense layer functions

`int layer_init(Layer *layer, int input_size, int output_size, Activation activation)`
- Initializes a dense layer and allocates internal buffers (`weights`, `biases`, forward caches).
- Use `layer_free` to release resources.

`void layer_free(Layer *layer)`
- Frees memory owned by a dense `Layer`.

`int layer_forward(Layer *layer, const float *input, float *output)`
- Runs dense forward pass: `output = activation(W * input + b)`.
- Caches input and pre-activation values for backward pass.

`int layer_backward(const Layer *layer, const float *delta_in, float *delta_out, float *grad_w, float *grad_b)`
- Runs backward pass for one dense layer.
- Produces parameter gradients (`grad_w`, `grad_b`) and optional propagated delta (`delta_out`).

### Conv2D functions

`int conv2d_layer_init(Conv2DLayer *layer, int input_width, int input_height, int input_channels, int output_channels, int kernel_width, int kernel_height, int stride, int padding, Activation activation)`
- Initializes a Conv2D layer, validates geometry, computes output shape, and allocates parameters/caches.

`void conv2d_layer_free(Conv2DLayer *layer)`
- Frees memory owned by a Conv2D layer.

`int conv2d_layer_forward(Conv2DLayer *layer, const float *input, float *output)`
- Runs Conv2D forward pass on flattened CHW input and output tensors.

`int conv2d_layer_backward(const Conv2DLayer *layer, const float *delta_in, float *delta_out, float *grad_w, float *grad_b)`
- Runs Conv2D backward pass.
- Computes kernel/bias gradients and optional propagated input delta.

### MaxPool2D functions

`int maxpool2d_layer_init(MaxPool2DLayer *layer, int input_width, int input_height, int channels, int pool_width, int pool_height, int stride, int padding)`
- Initializes a MaxPool2D layer, validates geometry, computes output shape, and allocates caches.

`void maxpool2d_layer_free(MaxPool2DLayer *layer)`
- Frees memory owned by a MaxPool2D layer.

`int maxpool2d_layer_forward(MaxPool2DLayer *layer, const float *input, float *output)`
- Runs MaxPool2D forward pass on flattened CHW tensors.
- Stores max indices for use in backward pass.

`int maxpool2d_layer_backward(const MaxPool2DLayer *layer, const float *delta_in, float *delta_out, float *grad_w, float *grad_b)`
- Routes each pooled gradient back to the winning input index from forward pass.
- `grad_w`/`grad_b` are dummy outputs (kept for plugin API compatibility).

## `layers.h`

### Layer plugin constructors

`int layer_plugin_dense_create(int input_size, int output_size, Activation activation, LayerPlugin *out_plugin)`
- Creates a dense layer plugin and fills callback pointers in `out_plugin`.
- Ownership of the internal context is transferred to the plugin.

`int layer_plugin_conv2d_create(int input_width, int input_height, int input_channels, int output_channels, int kernel_width, int kernel_height, int stride, int padding, Activation activation, LayerPlugin *out_plugin)`
- Creates a Conv2D plugin for flattened CHW tensors.

`int layer_plugin_maxpool2d_create(int input_width, int input_height, int channels, int pool_width, int pool_height, int stride, int padding, LayerPlugin *out_plugin)`
- Creates a MaxPool2D plugin for flattened CHW tensors.

`void layer_plugin_free(LayerPlugin *plugin)`
- Frees plugin-owned context via `destroy` callback and resets plugin pointers.

## `lossfunctions.h`

`float loss_mse(const float *pred, const float *target, int size)`
- Computes mean squared error between prediction and target vectors.

`int loss_mse_grad(const float *pred, const float *target, int size, float *grad_out)`
- Computes gradient of MSE with respect to predictions into `grad_out`.

`float loss_bce(const float *pred, const float *target, int size)`
- Computes binary cross-entropy between prediction and target vectors.

`int loss_bce_grad(const float *pred, const float *target, int size, float *grad_out)`
- Computes gradient of BCE with respect to predictions into `grad_out`.

## `optimizers.h`

`int adam_optimizer(float *weights, float *grads, float *m, float *v, float beta1, float beta2, float learning_rate, int t, int size)`
- Applies one in-place Adam update to `weights` using gradients and moment buffers.
- `m` and `v` are updated in-place.

`int sgd_optimizer(float *weights, float *grads, float learning_rate, int size)`
- Applies one in-place SGD update to `weights`.

`int rmsprop_optimizer(float *weights, float *grads, float *cache, float beta, float learning_rate, int size)`
- Applies one in-place RMSProp update to `weights` using running `cache`.

## `models.h`

### Model and training state structs

`AdamOptimizerState`
- Holds per-layer optimizer buffers (`m_w`, `v_w`, `m_b`, `v_b`), step counter, and betas.
- Used directly for Adam and as cache container for RMSProp helpers.

`SequentialModel`
- Dynamic list of `LayerPlugin` layers and optional compiled training configuration.

`SequentialTrainConfig`
- Compact runtime config for one-step training helper (`loss`, `optimizer`, `lr`, optional state).

### Sequential model lifecycle

`int sequential_model_init(SequentialModel *model, int initial_capacity)`
- Initializes a dynamic sequential model container with starting layer capacity.

`void sequential_model_free(SequentialModel *model)`
- Frees all layers and internal model resources.

### Add layers to sequential model

`int sequential_model_add_layer(SequentialModel *model, LayerPlugin layer)`
- Appends an already-created plugin layer to the model.
- Transfers ownership of `layer` into `model`.

`int sequential_model_add_dense(SequentialModel *model, int input_size, int output_size, Activation activation)`
- Convenience helper that creates and appends a dense plugin layer.

`int sequential_model_add_conv2d(SequentialModel *model, int input_width, int input_height, int input_channels, int output_channels, int kernel_width, int kernel_height, int stride, int padding, Activation activation)`
- Convenience helper that creates and appends a Conv2D plugin layer.

`int sequential_model_add_maxpool2d(SequentialModel *model, int input_width, int input_height, int channels, int pool_width, int pool_height, int stride, int padding)`
- Convenience helper that creates and appends a MaxPool2D plugin layer.

### Inference and initialization

`int sequential_model_forward(SequentialModel *model, const float *input, float *output)`
- Runs forward pass through all model layers.

`int sequential_model_randomize(SequentialModel *model, float init_scale)`
- Random-initializes all layer weights and biases in range `[-init_scale/2, +init_scale/2]`.

`int sequential_model_predict(SequentialModel *model, const float *input, float *output)`
- Alias of `sequential_model_forward`.

### Save / load `.lnn`

`int sequential_model_save_lnn(const SequentialModel *model, const char *file_path)`
- Saves model parameters (weights and biases) to a binary `.lnn` file.
- File extension must be `.lnn`.

`int sequential_model_load_lnn(SequentialModel *model, const char *file_path)`
- Loads model parameters from a binary `.lnn` file into an existing model.
- The current model architecture must match the file (layer count and parameter sizes).

### Compile + fit API

`int sequential_model_compile(SequentialModel *model, LossFunctionType loss_function, OptimizerType optimizer, float learning_rate, float adam_beta1, float adam_beta2)`
- Stores training settings in the model.
- Initializes internal optimizer state for Adam/RMSProp.

`int sequential_model_train(SequentialModel *model, const float *inputs, const float *targets, int num_samples, int input_size, int target_size, int epochs, float *final_loss_out)`
- Trains for `epochs` over contiguous sample arrays using compiled settings.
- Writes final average loss to `final_loss_out` when provided.

### Config helpers

`void sequential_train_config_init_sgd(SequentialTrainConfig *cfg, LossFunctionType loss_function, float learning_rate)`
- Fills config for SGD training steps.

`void sequential_train_config_init_adam(SequentialTrainConfig *cfg, LossFunctionType loss_function, float learning_rate, AdamOptimizerState *adam_state)`
- Fills config for Adam training steps.

### External Adam state management

`int sequential_model_adam_state_init(SequentialModel *model, AdamOptimizerState *out_state, float beta1, float beta2)`
- Allocates optimizer buffers for each layer in `model` and initializes Adam settings.

`void sequential_model_adam_state_free(SequentialModel *model, AdamOptimizerState *state)`
- Frees buffers allocated by `sequential_model_adam_state_init`.

### One-step training (plugin sequential API)

`int sequential_model_train_step_cfg(SequentialModel *model, const float *input, const float *target, float *output, const SequentialTrainConfig *cfg, float *loss_out)`
- Executes one train step using compact `cfg` (forward + loss grad + backward + update).

`int sequential_model_train_step(SequentialModel *model, const float *input, const float *target, float *output, LossFunctionType loss_function, OptimizerType optimizer, float learning_rate, AdamOptimizerState *adam_state, float *loss_out)`
- One training step with selectable loss (`LOSS_MSE` or `LOSS_BCE`) and optimizer.

### Optimize from already-computed prediction (plugin sequential API)

`int sequential_model_optimize_from_prediction(SequentialModel *model, const float *prediction, const float *target, LossFunctionType loss_function, OptimizerType optimizer, float learning_rate, AdamOptimizerState *adam_state, float *loss_out)`
- Runs optimization phase only using existing `prediction` from a previous forward pass. Pass `LOSS_MSE` or `LOSS_BCE`.

### Layer-array sequential API

`int sequential_forward(Layer *layers, int num_layers, const float *input, float *output)`
- Runs forward pass through a raw array of dense `Layer` structs.

`int sequential_train_step_sgd(Layer *layers, int num_layers, const float *input, const float *target, float *output, float **grads_w, float **grads_b, float learning_rate, float *loss_out)`
`int sequential_train_step(Layer *layers, int num_layers, const float *input, const float *target, float *output, float **grads_w, float **grads_b, LossFunctionType loss_function, OptimizerType optimizer, float learning_rate, AdamOptimizerState *adam_state, float *loss_out)`
- One-step training on dense layer arrays with selectable loss and optimizer.

`int sequential_optimize_from_prediction(Layer *layers, int num_layers, const float *prediction, const float *target, float **grads_w, float **grads_b, LossFunctionType loss_function, OptimizerType optimizer, float learning_rate, AdamOptimizerState *adam_state, float *loss_out)`
- Optimization-only helper for layer arrays with selectable loss and optimizer.

## Common Error Cases

Most `-1` returns are caused by:
- `NULL` pointers in required arguments.
- Non-positive sizes or invalid geometry.
- Architecture mismatch between serialized `.lnn` data and target model.
- Missing optimizer state for Adam/RMSProp training helpers.
- Memory allocation or file I/O failure.

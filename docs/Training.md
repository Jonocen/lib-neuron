# Training

This page explains the training flow in `lib-neuron`.

## Core pieces

- `matrixcalculation`: `layer_forward`, `layer_backward`
- `lossfunctions`: `loss_mse`, `loss_mse_grad`, `loss_bce`, `loss_bce_grad`
- `optimizers`: `sgd_optimizer`, `adam_optimizer`, `rmsprop_optimizer`
- `models`: sequential helpers

Loss selection in models:

- `sequential_model_train_step_mse` / `sequential_model_train_step_bce`
- `sequential_train_step_mse` / `sequential_train_step_bce`
- `sequential_model_train_step_with_loss` / `sequential_train_step_with_loss`

Easy sequential workflow:

- `sequential_model_adam_state_init`
- `sequential_train_config_init_adam` or `sequential_train_config_init_sgd`
- `sequential_model_train_step_cfg`

Framework-style workflow:

- `sequential_model_compile`
- `sequential_model_train`
- `sequential_model_predict`

## Built-in sequential training

Use `sequential_train_step` (array-of-layers API) or `sequential_model_train_step` (plugin API).

These default to `LOSS_MSE` for compatibility.

If you want explicit loss naming, use the `_mse` and `_bce` variants.

If you want runtime loss selection, use `*_with_loss` variants.

One-call helper pattern:

```c
sequential_model_train_step(&model,
							input,
							target,
							output,
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

Compile/fit/predict pattern:

```c
sequential_model_compile(&model, LOSS_MSE, OPTIMIZER_SGD, 0.05f, 0.9f, 0.999f);
sequential_model_train(&model, &x[0][0], &y[0][0], 4, 2, 1, 5000, &loss);
sequential_model_predict(&model, input, output);
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
										  OPTIMIZER_ADAM,
										  learning_rate,
										  &adam,
										  NULL);
```

Split pattern with explicit BCE:

```c
sequential_model_forward(&model, input, output);

sequential_model_optimize_from_prediction_bce(&model,
											  output,
											  target,
											  OPTIMIZER_ADAM,
											  learning_rate,
											  &adam,
											  &loss);
```

Both do this sequence each step:

1. Forward pass
2. Compute selected loss (MSE or BCE)
3. Compute output gradient
4. Backpropagate layer-by-layer
5. Update weights/biases with the selected optimizer

SGD compatibility wrappers still exist:

- `sequential_train_step_sgd`
- `sequential_model_train_step_sgd`

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
					  OPTIMIZER_ADAM,
					  learning_rate,
					  &adam,
					  &loss);
```

For stable training, keep `m`, `v`, and `t` persistent across all epochs/batches.

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
- If training stalls near 0.5 predictions, test different seeds/init scale.
- BCE can work better than MSE for sigmoid-based binary outputs.

# Training

This page explains the training flow in `lib-neuron`.

## Core pieces

- `matrixcalculation`: `layer_forward`, `layer_backward`
- `lossfunctions`: `loss_mse`, `loss_mse_grad`, `loss_bce`, `loss_bce_grad`
- `optimizers`: `sgd_optimizer`, `adam_optimizer`
- `models`: sequential helpers

## Built-in sequential training

Use `sequential_train_step_sgd` (array-of-layers API) or `sequential_model_train_step_sgd` (plugin API).

Both do this sequence each step:

1. Forward pass
2. Compute loss (MSE in current helper)
3. Compute output gradient
4. Backpropagate layer-by-layer
5. Update weights/biases with SGD

## Using Adam manually

`adam_optimizer` is available and can be used per parameter buffer.

Required Adam state per parameter:

- first moment `m`
- second moment `v`
- global step `t` (must start at 1)

Pseudo-usage:

```c
adam_optimizer(weights, grads, m, v, 0.9f, 0.999f, learning_rate, t, size);
```

For stable training, keep `m`, `v`, and `t` persistent across all epochs/batches.

## Practical tips

- Keep initialization small (for XOR, small random weights help convergence).
- If training stalls near 0.5 predictions, test different seeds/init scale.
- BCE can work better than MSE for sigmoid-based binary outputs.

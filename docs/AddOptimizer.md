# Add a New Optimizer (Step by Step)

This guide shows how to add a new optimizer end-to-end in `lib-neuron`.

Use this checklist in order.

## 1. Add the optimizer type

Edit `include/optimizers.h`.

- Add a new enum value in `OptimizerType`.
- Keep existing values unchanged if you want to preserve compatibility.

Example:

```c
typedef enum {
    OPTIMIZER_SGD = 0,
    OPTIMIZER_ADAM = 1,
    OPTIMIZER_RMSPROP = 2,
    OPTIMIZER_ADAGRAD = 3
} OptimizerType;
```

## 2. Add function declaration

Still in `include/optimizers.h`, declare the optimizer function.

Example:

```c
int adagrad_optimizer(float *weights,
                      float *grads,
                      float *accumulator,
                      float learning_rate,
                      int size);
```

## 3. Implement optimizer math

Edit `src/optimizers.c`.

- Add argument validation.
- Apply in-place updates to `weights`.
- Update state buffers (if needed).

Example skeleton:

```c
int adagrad_optimizer(float *weights,
                      float *grads,
                      float *accumulator,
                      float learning_rate,
                      int size) {
    if (!weights || !grads || !accumulator || size <= 0) return -1;
    if (learning_rate <= 0.0f) return -1;

    for (int i = 0; i < size; i++) {
        accumulator[i] += grads[i] * grads[i];
        weights[i] -= learning_rate * grads[i] / (sqrtf(accumulator[i]) + 1e-8f);
    }

    return 0;
}
```

## 4. Decide optimizer state shape

Edit `include/models.h` and use `OptimizerState` fields.

Current state fields:

- `m_w`, `m_b`
- `v_w`, `v_b`
- `step`, `beta1`, `beta2`

For new optimizers, map these fields clearly.

Example mapping for Adagrad:

- `m_w` and `m_b`: accumulators
- `v_w` and `v_b`: unused
- `step`: unused
- `beta1`, `beta2`: unused

If you need new state values, add them to `OptimizerState` carefully and update all init/free paths.

## 5. Wire update path in models

Edit `src/models.c`.

### 5.1 Update state validation

Add your optimizer branch in `optimizer_state_valid(...)`.

- Return `1` for valid state, `0` for invalid.
- For stateless optimizer, allow `NULL` state.

### 5.2 Update optimizer dispatch

In `apply_optimizer_update(...)`, add a branch for your optimizer.

Example pattern:

```c
if (optimizer == OPTIMIZER_ADAGRAD) {
    if (!optimizer_state || !opt_state_a) return -1;
    return adagrad_optimizer(weights,
                             (float *)grads,
                             opt_state_a,
                             learning_rate,
                             size);
}
```

### 5.3 Select per-layer state pointers

In training loops (`sequential_model_train_with_progress`,
`sequential_model_optimize_from_prediction`, and
`sequential_optimize_from_prediction`), add pointer wiring similar to Adam/RMSProp:

- Choose the correct state arrays per layer.
- Pass them to `apply_optimizer_update(...)`.

## 6. Initialize optimizer state

Use `sequential_model_optimizer_state_init(...)` logic in `src/models.c`.

- Add your optimizer-specific validation.
- Allocate only the buffers your optimizer needs.
- Keep zero-initialized state.

Also ensure cleanup works in:

- `sequential_model_optimizer_state_free(...)`
- compile reconfigure paths that free/re-init state

## 7. Compile-time integration

Ensure compile APIs accept your optimizer:

- `sequential_model_compile(...)`
- `sequential_model_compile_optimizer(...)`

Checklist:

- optimizer enum accepted
- required hyperparameter checks added
- internal state initialized for your optimizer when needed

## 8. Config helper integration

If helpful, add a config helper in `include/models.h` and `src/models.c`.

Example:

```c
void sequential_train_config_init_adagrad(SequentialTrainConfig *cfg,
                                          LossFunctionType loss_function,
                                          float learning_rate,
                                          OptimizerState *optimizer_state);
```

This keeps training call sites simple.

## 9. Documentation updates

Update docs so users can discover and use the new optimizer.

Minimum files:

- `docs/Training.md`
- `docs/APIReference.md`
- this file (`docs/AddOptimizer.md`) if steps changed

## 10. Sanity checks

Run these before merging:

1. `make`
2. Train a tiny model with the new optimizer.
3. Verify loss decreases on a simple dataset.
4. Test failure behavior with invalid args (expects `-1`).

## 11. Minimal usage example

```c
OptimizerState state = {0};
float loss = 0.0f;

sequential_model_optimizer_state_init(&model,
                                      &state,
                                      OPTIMIZER_ADAGRAD,
                                      0.9f,
                                      0.999f);

sequential_model_train_step(&model,
                            input,
                            target,
                            output,
                            LOSS_MSE,
                            OPTIMIZER_ADAGRAD,
                            0.01f,
                            &state,
                            &loss);

sequential_model_optimizer_state_free(&model, &state);
```

Note:
- The `beta1/beta2` parameters are currently shared generic parameters.
- If your optimizer does not use them, document that clearly and ignore them safely.

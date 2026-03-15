# Add a New Optimizer (Beginner Guide)

This guide is for beginners who want to add a new optimizer safely in `lib-neuron`.

Goal:
- Add one new optimizer (example: `Adagrad`)
- Make it usable in all training APIs
- Keep project behavior consistent (`0` success, `-1` failure)

You can follow this exactly, top to bottom.

## Before You Start

You will edit these files:

1. `include/optimizers.h`
2. `src/optimizers.c`
3. `src/models_internal.c`
4. `src/models_train.c`
5. `src/models_state.c`
6. `src/models_legacy.c`
7. `docs/Training.md`
8. `docs/APIReference.md`

Run this after each big step:

```sh
make
```

## Mental Model (Simple)

An optimizer in this project is:

1. A math function that updates weights from gradients.
2. A new enum value (`OptimizerType`) so code can select it.
3. A state mapping (if it needs memory across steps).
4. Wiring in models internals/train/state files so training loops call it.

If one of these is missing, the optimizer will not work end-to-end.

## Step 1. Add the enum + function declaration

Edit `include/optimizers.h`.

Add your optimizer enum value:

```c
typedef enum {
    OPTIMIZER_SGD = 0,
    OPTIMIZER_ADAM = 1,
    OPTIMIZER_RMSPROP = 2,
    OPTIMIZER_ADAGRAD = 3,
    OPTIMIZER_ADAMW = 4
} OptimizerType;
```

Add function declaration:

```c
int adagrad_optimizer(float *weights,
                      float *grads,
                      float *accumulator,
                      float learning_rate,
                      int size);
```

Why:
- Enum makes the optimizer selectable.
- Declaration lets other files call the function.

## Step 2. Implement optimizer math

Edit `src/optimizers.c` and add:

```c
int adagrad_optimizer(float *weights, float *grads, float *accumulator,
                      float learning_rate, int size) {
    if (!weights || !grads || !accumulator || size <= 0) return -1;
    if (learning_rate <= 0.0f) return -1;

    for (int i = 0; i < size; i++) {
        accumulator[i] += grads[i] * grads[i];
        weights[i] -= learning_rate * grads[i] / (sqrtf(accumulator[i]) + 1e-8f);
    }

    return 0;
}
```

Why:
- `accumulator` stores running squared gradients.
- It must persist between steps, so it is optimizer state.

## Step 3. Choose state mapping

`lib-neuron` already has `OptimizerState` in `include/models_types.h`.

For Adagrad, use:

- `m_w` and `m_b` as accumulators
- `v_w` and `v_b` unused
- `step` unused
- `beta1/beta2` unused

Code snippet (mapping idea):

```c
/* Adagrad state mapping in this project:
 * - m_w / m_b: accumulators
 * - v_w / v_b: unused
 * - step: unused
 * - beta1 / beta2: unused
 */
static int adagrad_optimizer_state_valid(const OptimizerState *optimizer_state) {
    if (!optimizer_state) return 0;
    if (!optimizer_state->m_w || !optimizer_state->m_b) return 0;
    return 1;
}
```

This avoids adding a new struct and keeps code simpler.

## Step 4. Validate state in `src/models_internal.c`

Add a helper similar to existing ones:

```c
static int adagrad_optimizer_state_valid(const OptimizerState *optimizer_state) {
    if (!optimizer_state) return 0;
    if (!optimizer_state->m_w || !optimizer_state->m_b) return 0;
    return 1;
}
```

Then extend `optimizer_state_valid(...)`:

```c
if (optimizer == OPTIMIZER_ADAGRAD) {
    return adagrad_optimizer_state_valid(optimizer_state);
}
```

Why:
- Training should fail early if required state is missing.

## Step 5. Add dispatch in `lnn_apply_optimizer_update(...)`

In `src/models_internal.c`, extend optimizer dispatch:

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

Why:
- This is the central place where optimizers are called.

## Step 6. Allow compile/config paths to accept new optimizer

In `src/models_train.c` and `src/models_state.c`, update checks that list allowed optimizers.

Typical places:

- `sequential_model_compile_optimizer(...)`
- `sequential_model_optimizer_state_init(...)`

Add `OPTIMIZER_ADAGRAD` to the allowed set.

Then ensure compile-time state ownership includes Adagrad, same style as Adam/RMSProp.

## Step 7. Wire per-layer state pointers in training loops

In these functions:

1. `sequential_model_train_with_progress(...)`
2. `sequential_model_optimize_from_prediction(...)`
3. `sequential_optimize_from_prediction(...)`

These live in:

- `src/models_train.c`
- `src/models_state.c`
- `src/models_legacy.c`

When optimizer is Adagrad, set:

- `opt_state_w_a = optimizer_state->m_w[l];`
- `opt_state_b_a = optimizer_state->m_b[l];`

Why:
- Adagrad needs one accumulator per parameter.

## Step 8. Keep state init/free compatible

In `sequential_model_optimizer_state_init(...)`:

- Accept `OPTIMIZER_ADAGRAD`.
- Allocate `m_w` and `m_b` arrays.
- `v_w`/`v_b` may remain unused for Adagrad.

In `sequential_model_optimizer_state_free(...)`:

- No special change if all arrays are already freed generically.

## Step 9. Update docs

Update:

1. `docs/Training.md`
2. `docs/APIReference.md`

Add:

- enum value `OPTIMIZER_ADAGRAD`
- function `adagrad_optimizer(...)`
- brief usage guidance (typical learning rates)

## Step 10. Beginner test checklist

Run all of these:

1. `make`
2. Train a tiny model with `OPTIMIZER_ADAGRAD`
3. Confirm loss goes down
4. Try invalid input (like `NULL` accumulator) and confirm `-1`

## Copy-Paste Example Usage

```c
OptimizerState state = {0};
float loss = 0.0f;

/* For Adagrad, beta params are ignored by current implementation. */
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

## Common Mistakes

1. Added enum, but forgot to add branch in `apply_optimizer_update(...)`.
2. Added update branch, but forgot training-loop state pointer wiring.
3. Forgot to allow optimizer in compile/validation checks.
4. Forgot docs update, so users cannot discover the feature.

If you hit one of these, the optimizer often compiles but fails at runtime with `-1`.

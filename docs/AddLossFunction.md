# Add a New Loss Function (Beginner Guide)

This guide is for beginners who want to add a new loss function safely in `lib-neuron`.

Goal:
- Add one new loss (example: `Huber`)
- Make it work in all training APIs
- Keep project behavior consistent (`0` success, `-1` failure)

## Before You Start

You will edit these files:

1. `include/lossfunctions.h`
2. `src/lossfunctions.c`
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

A loss in this project has 2 parts:

1. A value function (example: `loss_huber`) that returns a single float.
2. A gradient function (example: `loss_huber_grad`) that fills `grad_out`.

If you only add one of these, training will not work.

## Step 1. Add enum value

Edit `include/lossfunctions.h` and add your new enum value.

```c
typedef enum {
    LOSS_MSE = 0,
    LOSS_BCE = 1,
    LOSS_HUBER = 2
} LossFunctionType;
```

If not already there:

- Add `LOSS_HUBER = 2` (or the next free value).
- Save and run `make`.

## Step 2. Add declarations

Still in `include/lossfunctions.h`, add both declarations:

```c
float loss_huber(const float *pred, const float *target, int size, float delta);
int loss_huber_grad(const float *pred, const float *target, int size, float delta, float *grad_out);
```

If not already there:

- Add both lines.
- If only one exists, you will get compile/link errors later.

## Step 3. Implement the math

Edit `src/lossfunctions.c` and add both function definitions.

Copy this starter:

```c
float loss_huber(const float *pred, const float *target, int size, float delta) {
    if (!pred || !target || size <= 0 || delta <= 0.0f) return -1.0f;

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        float e = pred[i] - target[i];
        float a = fabsf(e);
        if (a <= delta) {
            sum += 0.5f * e * e;
        } else {
            sum += delta * (a - 0.5f * delta);
        }
    }

    return sum / (float)size;
}

int loss_huber_grad(const float *pred, const float *target, int size, float delta, float *grad_out) {
    if (!pred || !target || !grad_out || size <= 0 || delta <= 0.0f) return -1;

    for (int i = 0; i < size; i++) {
        float e = pred[i] - target[i];
        float a = fabsf(e);
        grad_out[i] = (a <= delta)
            ? (e / (float)size)
            : (((e > 0.0f) ? delta : -delta) / (float)size);
    }

    return 0;
}
```

Checklist:

- `#include <math.h>` exists.
- Value function returns `-1.0f` on invalid input.
- Gradient function returns `-1` on invalid input.

## Step 4. Wire into training flow

Edit `src/models_internal.c` in `lnn_compute_loss_and_grad(...)`.

Add a `LOSS_HUBER` branch:

```c
if (loss_function == LOSS_HUBER) {
    if (loss_out) {
        *loss_out = loss_huber(prediction, target, size, 1.0f);
    }
    return loss_huber_grad(prediction, target, size, 1.0f, grad_out);
}
```

Notes:

- `1.0f` is a default delta.
- If you want configurable delta, add it to API later.

## Step 5. Allow Huber in compile validation

In `src/models_train.c`, find loss checks like this:

```c
if (loss_function != LOSS_MSE && loss_function != LOSS_BCE) return -1;
```

Update to:

```c
if (loss_function != LOSS_MSE &&
    loss_function != LOSS_BCE &&
    loss_function != LOSS_HUBER) return -1;
```

If not already there:

- Update `sequential_model_compile_optimizer(...)` first.

## Step 6. Confirm all training APIs are covered

You do not need to edit every train function if they all call `compute_loss_and_grad(...)`.

That single shared function powers:

- `sequential_model_train_step(...)`
- `sequential_model_optimize_from_prediction(...)`
- `sequential_train_step(...)`
- `sequential_optimize_from_prediction(...)`
- `sequential_model_train(...)`
- `sequential_model_train_with_progress(...)`

## Step 7. Update docs

Update at least:

1. `docs/Training.md`
2. `docs/APIReference.md`
3. this file (`docs/AddLossFunction.md`) when process changes

Add:

- `LOSS_HUBER` enum entry
- `loss_huber(...)` and `loss_huber_grad(...)` API entries

## Step 8. Beginner test checklist

Run this:

1. `make`
2. Train a tiny model with `LOSS_HUBER`
3. Confirm loss decreases
4. Confirm invalid args return `-1` / `-1.0f`
5. Confirm gradient has no NaN/Inf

## Copy-Paste Example Usage

```c
sequential_model_compile(&model,
                         LOSS_HUBER,
                         OPTIMIZER_SGD,
                         0.01f,
                         0.9f,
                         0.999f);

sequential_model_train(&model,
                       inputs,
                       targets,
                       num_samples,
                       input_size,
                       target_size,
                       200,
                       8,
                       &final_loss);
```

## Quick Troubleshooting

- `unknown identifier LOSS_HUBER`:
  Step 1 not done, or old header is being used.

- `undefined reference to loss_huber`:
  Step 3 implementation missing in `src/lossfunctions.c`.

- Training returns `-1` when using Huber:
  Step 5 validation check not updated.

- Build passes but no learning:
  Check gradient scale and learning rate.

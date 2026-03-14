# Add a New Loss Function (Step by Step)

This guide shows how to add a new loss function end-to-end in `lib-neuron`.

Use this checklist in order.

## 1. Add loss enum value

Edit `include/lossfunctions.h`.

- Add a new value in `LossFunctionType`.
- Keep existing values stable if you care about compatibility.

Example:

```c
typedef enum {
    LOSS_MSE = 0,
    LOSS_BCE = 1,
    LOSS_HUBER = 2
} LossFunctionType;
```

## 2. Add function declarations

Still in `include/lossfunctions.h`, add both value and gradient functions.

Example:

```c
float loss_huber(const float *pred, const float *target, int size, float delta);
int loss_huber_grad(const float *pred, const float *target, int size, float delta, float *grad_out);
```

Important:
- Every new loss must have a gradient function because training uses backpropagation from loss gradients.

## 3. Implement math in `src/lossfunctions.c`

Add both functions with input validation and numerically safe behavior.

General rules:

- Return `-1.0f` for invalid args in value functions.
- Return `-1` for invalid args in gradient functions.
- Keep gradient shape exactly equal to prediction shape (`size`).

Example skeleton:

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
        grad_out[i] = (a <= delta) ? (e / (float)size)
                                   : ((e > 0 ? delta : -delta) / (float)size);
    }
    return 0;
}
```

## 4. Wire the loss into model training

Edit `src/models.c` in `compute_loss_and_grad(...)`.

Add a branch for your new enum:

- compute scalar loss into `loss_out` when not `NULL`
- fill `grad_out` using your gradient function
- return `0` on success or `-1` on failure

Example pattern:

```c
if (loss_function == LOSS_HUBER) {
    if (loss_out) {
        *loss_out = loss_huber(prediction, target, size, 1.0f);
    }
    return loss_huber_grad(prediction, target, size, 1.0f, grad_out);
}
```

## 5. Update validation checks

In `src/models.c`, update checks that currently allow only `LOSS_MSE` and `LOSS_BCE`.

Most important place:

- `sequential_model_compile(...)` (or `sequential_model_compile_optimizer(...)` if using generic API)

Make sure your new loss is accepted.

## 6. Keep API behavior consistent

Your loss should work in all training entry points automatically once step 4 is done:

- `sequential_model_train_step(...)`
- `sequential_model_optimize_from_prediction(...)`
- `sequential_train_step(...)`
- `sequential_optimize_from_prediction(...)`
- `sequential_model_train(...)`
- `sequential_model_train_with_progress(...)`

Reason: these flows all pass through shared loss/gradient logic.

## 7. Add docs

Update at least:

- `docs/Training.md`
- `docs/APIReference.md`
- this file (`docs/AddLossFunction.md`) when process changes

## 8. Sanity checks

Run before merging:

1. `make`
2. Small training run with the new loss.
3. Verify loss decreases.
4. Verify invalid arguments return `-1` / `-1.0f`.
5. Verify gradient has no NaN/Inf on normal inputs.

## 9. Minimal usage example

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

## 10. Common pitfalls

- Forgetting to add the gradient function.
- Adding enum and implementation, but not updating compile-time loss validation.
- Not protecting against numerical issues (for example `log(0)` in BCE-style losses).
- Returning a wrong gradient scale (missing divide by `size` if your other losses are averaged).

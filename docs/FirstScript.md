# Use the Library + Your First Script

This guide shows the smallest path from zero to a running program with `lib-neuron`.

## 1) Build the library

From project root:

```sh
make
```

This creates `libneuron.a`.

## 2) Create your first script

Create a file, for example `my_first_nn.c`:

```c
#include <lib-neuron.h>
#include <stdio.h>

int main(void) {
    SequentialModel model;
    float input[2] = {1.0f, 0.0f};
    float target[1] = {1.0f};
    float output[1];
    float loss = 0.0f;

    if (sequential_model_init(&model, 2) != 0) return 1;
    if (sequential_model_add_dense(&model, 2, 4, ACT_RELU) != 0) return 1;
    if (sequential_model_add_dense(&model, 4, 1, ACT_SIGMOID) != 0) return 1;

    /* One training step */
    if (sequential_model_train_step_sgd(&model, input, target, output, 0.05f, &loss) != 0) {
        sequential_model_free(&model);
        return 1;
    }

    printf("loss=%f, out=%f\n", loss, output[0]);

    /* Inference */
    if (sequential_model_forward(&model, input, output) != 0) {
        sequential_model_free(&model);
        return 1;
    }
    printf("prediction=%f\n", output[0]);

    sequential_model_free(&model);
    return 0;
}
```

## 3) Compile your script

From project root:

```sh
gcc my_first_nn.c -Iinclude -L. -lneuron -lm -o my_first_nn
```

## 4) Run

```sh
./my_first_nn
```

## What to try next

- Train in a loop over a dataset (like XOR examples).
- Try `ACT_TANH` or `ACT_RELU` in hidden layers.
- Use `loss_bce` / `loss_bce_grad` and `adam_optimizer` manually for binary tasks.
- Check full working examples in `examples/`.

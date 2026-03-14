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
    float input[1][2] = {{1.0f, 0.0f}};
    float target[1][1] = {{1.0f}};
    float output[1];
    float loss = 0.0f;

    if (sequential_model_init(&model, 2) != 0) return 1;
    if (sequential_model_add_dense(&model, 2, 4, ACT_RELU) != 0) return 1;
    if (sequential_model_add_dense(&model, 4, 1, ACT_SIGMOID) != 0) return 1;

    if (sequential_model_compile(&model,
                                 LOSS_BCE,
                                 OPTIMIZER_SGD,
                                 0.05f,
                                 0.9f,
                                 0.999f) != 0) {
        sequential_model_free(&model);
        return 1;
    }

    if (sequential_model_train(&model,
                               &input[0][0],
                               &target[0][0],
                               1,
                               2,
                               1,
                               10,
                               &loss) != 0) {
        sequential_model_free(&model);
        return 1;
    }
    printf("loss=%f, out=%f\n", loss, output[0]);

    /* Inference */
    if (sequential_model_predict(&model, input[0], output) != 0) {
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
- Switch optimizer with `OPTIMIZER_SGD` / `OPTIMIZER_ADAM`.
- Switch loss with `LOSS_MSE` / `LOSS_BCE`.
- Use `sequential_model_compile` + `sequential_model_train` for the shortest training flow.
- Check full working examples in `examples/`.

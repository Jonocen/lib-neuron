# Examples

Current examples are in `examples/`.

## Build all examples

```sh
make examples
```

## Example 1: classic XOR training

Source: `examples/Other_Exaple.c`

Build:

```sh
make Other_Exaple
```

Run:

```sh
./examples/Other_Exaple
```

## Example 2: plugin-based sequential XOR

Source: `examples/sequential_xor_plugin.c`

Build:

```sh
make sequential_xor_plugin
```

Run:

```sh
./examples/sequential_xor_plugin
```

What it demonstrates:

- dynamic `SequentialModel`
- plugin dense layers via `sequential_model_add_dense`
- training via `sequential_model_train_step_sgd`
- inference via `sequential_model_forward`

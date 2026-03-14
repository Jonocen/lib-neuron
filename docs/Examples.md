# Examples

Current examples are in `examples/`.

## Build all examples

```sh
make examples
```

## Example 0: simple and compact XOR

Source: `examples/simple_compact.c`

Build:

```sh
make simple_compact
```

Run:

```sh
./examples/simple_compact
```

What it demonstrates:

- minimal `SequentialModel` setup
- one-call training via `sequential_model_train_step_with_loss`
- selectable optimizer/loss with compact code

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

What it demonstrates:

- layer-array API (`Layer layers[]`)
- compact training loop with stack-allocated gradients
- loss/optimizer choice through `sequential_train_step_with_loss`
- low-level layer-array workflow in minimal lines

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
- framework-style flow: `sequential_model_compile` + `sequential_model_train` + `sequential_model_predict`

# Quickstart

This quickstart shows how to build the library and run both shipped examples.

## 1) Build the library

From project root:

```sh
make
```

This produces `libneuron.a`.

## 2) Build all examples from root

```sh
make examples
```

This builds:

- `examples/simple_compact`
- `examples/sequential_xor_plugin`
- `examples/Other_Exaple`

## 3) Run the examples

```sh
./examples/simple_compact
./examples/sequential_xor_plugin
./examples/Other_Exaple
```

`Other_Exaple` is a compact layer-array training example.

## 4) Compile your own program

```sh
gcc your_program.c -Iinclude -L. -lneuron -lm
```

Include all public APIs with:

```c
#include <lib-neuron.h>
```

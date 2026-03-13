# Quickstart

This quickstart shows how to build the library and compile one example.

## 1) Build the library

From project root:

```sh
make
```

This produces `libneuron.a`.

## 2) Build an example from root

```sh
make sequential_xor_plugin
```

This builds `examples/sequential_xor_plugin`.

## 3) Run the example

```sh
./examples/sequential_xor_plugin
```

## 4) Compile your own program

```sh
gcc your_program.c -Iinclude -L. -lneuron -lm
```

Include all public APIs with:

```c
#include <lib-neuron.h>
```

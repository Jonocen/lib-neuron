# Documentation

Welcome to the `lib-neuron` docs.

These docs cover dense, conv2d, and maxpool2d workflows across both low-level and sequential APIs.
They also cover image data helpers for loading PGM files and training from in-memory image datasets.

## Topics

- [Quickstart](Quickstart.md)
- [API Reference](APIReference.md)
- [Examples](Examples.md)
- [Training](Training.md)
- [Add an Optimizer](AddOptimizer.md)
- [Add a Loss Function](AddLossFunction.md)
- [Add a Layer + Plugin Layer](AddLayerAndPluginLayer.md)
- [Use the Library + First Script](FirstScript.md)

## Notes

- API returns `0` on success and `-1` on invalid input/failure.
- Most arrays are updated in-place through pointers.
- Tensor layout for conv/pool APIs is flattened CHW (channel-major).
- Model API is split across `models_types.h`, `models_core.h`, `models_training.h`, and `models_legacy.h`; `models.h` includes all of them.
- `examples/Other_Exaple.c` is a compact layer-array training demo (name kept as-is to match the current filename/target).
- `examples/mnist_tiny_pgm.c` demonstrates out-of-box image training with a conv/pool + flatten stack.
- `APIReference.md` is a complete function-by-function reference for all public headers.

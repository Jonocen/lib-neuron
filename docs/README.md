# Documentation

Welcome to the `lib-neuron` docs.

These docs cover dense, conv2d, and maxpool2d workflows across both low-level and sequential APIs.

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
- `examples/Other_Exaple.c` is a compact layer-array training demo (name kept as-is to match the current filename/target).
- `APIReference.md` is a complete function-by-function reference for all public headers.

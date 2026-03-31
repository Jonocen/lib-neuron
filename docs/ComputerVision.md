# Computer Vision

This page documents the `computervison` module for image loading and preprocessing in `lib-neuron`.

## Goal

Use image files as model input without external Python tooling:

- read image from disk
- optional resize
- optional grayscale/color conversion
- convert to flattened CHW float tensor

The resulting tensor is compatible with:

- `sequential_model_forward`
- `sequential_model_predict`
- `sequential_model_train_step`

## Supported formats

Current implementation supports:

- native `P5` (grayscale)
- native `P6` (RGB)
- fallback conversion for common formats (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tga`, `.gif`, `.webp`, ...)

Fallback conversion uses tools available in PATH: `magick`, `convert`, or `ffmpeg`.

## Core API

Include:

```c
#include <lib-neuron.h>
```

Useful functions:

- `jcv_imread`
- `jcv_resize`
- `jcv_convert_channels`
- `jcv_image_to_chw_float`
- `jcv_load_image_for_model`
- `jcv_format_image_for_model`
- `jcv_image_free`

## Data layouts

Image data in `JCVImage`:

- byte pixels
- HWC order (height, width, channel)

Model tensor output:

- float values
- CHW order (channel, height, width)
- flattened into a contiguous `float*`

For width `W`, height `H`, channels `C`:

- output size = `C * H * W`
- channel block size = `H * W`

Index mapping from HWC byte input to CHW float output:

- `src_idx = (y * W + x) * C + c`
- `dst_idx = c * (H * W) + (y * W + x)`

## Formatting helper

`jcv_format_image_for_model` lets you do all preprocessing in one call:

- force 1 or 3 channels
- optional resize
- optional RGB/BGR channel order
- optional normalization
- CHW float conversion

## Quick usage

```c
#include <stdio.h>
#include <stdlib.h>
#include <lib-neuron.h>

int main(void) {
    float *input = NULL;
    int input_size = 0;
    int channels = 0;

    if (jcv_load_image_for_model("sample.pgm",
                                 JCV_IMREAD_GRAYSCALE,
                                 28,
                                 28,
                                 JCV_INTER_LINEAR,
                                 1,
                                 &input,
                                 &input_size,
                                 &channels) != 0) {
        fprintf(stderr, "Image load failed\n");
        return 1;
    }

    printf("channels=%d size=%d\n", channels, input_size);

    free(input);
    return 0;
}
```

## Training flow

Typical classification flow:

1. Load each image using `jcv_load_image_for_model`.
2. Build target vector (one-hot or binary label).
3. Call `sequential_model_train_step` per sample or per mini-batch strategy.
4. Free each sample buffer with `free()`.

## Memory rules

- `jcv_load_image_for_model` allocates `out_input`; free it with `free()`.
- `jcv_imread` / `jcv_resize` / `jcv_convert_channels` allocate `JCVImage.data`; free with `jcv_image_free()`.
- Always handle `-1` return values for load/parse/alloc failures.

## Notes

- This module uses jcv-like naming and behavior but does not depend on OpenCV.
- `JCV_IMREAD_COLOR` always returns 3 channels.
- `JCV_IMREAD_GRAYSCALE` always returns 1 channel.
- Normalization is optional and controlled by `normalize_01`.
- Multi-format fallback depends on external converter binaries being installed.

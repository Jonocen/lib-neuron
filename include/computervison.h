#ifndef COMPUTERVISON_H
#define COMPUTERVISON_H

typedef enum {
    JCV_IMREAD_UNCHANGED = -1,
    JCV_IMREAD_GRAYSCALE = 0,
    JCV_IMREAD_COLOR = 1
} JCVImreadMode;

typedef enum {
    JCV_INTER_NEAREST = 0,
    JCV_INTER_LINEAR = 1
} JCVInterpolation;

typedef enum {
    JCV_CHANNEL_ORDER_RGB = 0,
    JCV_CHANNEL_ORDER_BGR = 1
} JCVChannelOrder;

typedef struct {
    int width;
    int height;
    int channels;
    unsigned char *data;
} JCVImage;

/*
 * Loads an image from disk.
 * - Native support: P5 (grayscale), P6 (RGB)
 * - Fallback support (if converter exists in PATH): jpg/jpeg/png/bmp/tga/gif/webp and more
 *   via `magick`, `convert`, or `ffmpeg` conversion to PPM.
 * - `mode` matches common jcv intent (unchanged, gray, color)
 * Returns 0 on success, -1 on failure.
 */
int jcv_imread(const char *file_path, JCVImreadMode mode, JCVImage *out_image);

/*
 * Resizes image using nearest-neighbor or bilinear interpolation.
 * Returns 0 on success, -1 on invalid input or allocation failure.
 */
int jcv_resize(const JCVImage *src,
               int new_width,
               int new_height,
               JCVInterpolation interpolation,
               JCVImage *out_image);

/*
 * Converts image channels to either 1 or 3 channels.
 * Returns 0 on success, -1 on failure.
 */
int jcv_convert_channels(const JCVImage *src, int out_channels, JCVImage *out_image);

/*
 * Converts image bytes to CHW float tensor (channel-first), suitable for
 * Conv2D/Sequential input in lib-neuron.
 * - If `normalize_01 != 0`, scales pixel values to [0, 1].
 * - `out_data` is heap allocated and must be freed with free().
 * Returns 0 on success, -1 on failure.
 */
int jcv_image_to_chw_float(const JCVImage *image,
                           int normalize_01,
                           float **out_data,
                           int *out_size);

/*
 * Convenience helper: load image, optional resize, convert to CHW float.
 * - If `target_width` and `target_height` are > 0, resizing is applied.
 * - `out_input` is heap allocated and must be freed with free().
 * Returns 0 on success, -1 on failure.
 */
int jcv_load_image_for_model(const char *file_path,
                             JCVImreadMode mode,
                             int target_width,
                             int target_height,
                             JCVInterpolation interpolation,
                             int normalize_01,
                             float **out_input,
                             int *out_size,
                             int *out_channels);

/*
 * Formats an in-memory image for model input in one step:
 * - optional channel conversion (1 or 3)
 * - optional resize
 * - optional RGB<->BGR reorder for 3-channel images
 * - HWC byte -> CHW float conversion
 * - optional [0,1] normalization
 * `out_input` is heap allocated and must be freed with free().
 * Returns 0 on success, -1 on failure.
 */
int jcv_format_image_for_model(const JCVImage *src,
                               int out_channels,
                               int target_width,
                               int target_height,
                               JCVInterpolation interpolation,
                               JCVChannelOrder channel_order,
                               int normalize_01,
                               float **out_input,
                               int *out_size);

/* Frees image data owned by `image`. Safe to call on zeroed structs. */
void jcv_image_free(JCVImage *image);

#endif /* COMPUTERVISON_H */

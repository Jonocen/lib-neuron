#ifndef LAYERS_H
#define LAYERS_H

#include "matrixcalculation.h"

typedef struct {
    void *ctx;

    int (*forward)(void *ctx, const float *input, float *output);
    int (*backward)(const void *ctx,
                    const float *delta_in,
                    float       *delta_out,
                    float       *grad_w,
                    float       *grad_b);

    int (*input_size)(const void *ctx);
    int (*output_size)(const void *ctx);

    float *(*weights)(void *ctx);
    float *(*biases)(void *ctx);
    int    (*weights_size)(const void *ctx);
    int    (*biases_size)(const void *ctx);

    void (*destroy)(void *ctx);
} LayerPlugin;

/*
 * Creates a dense (fully-connected) layer plugin backed by `Layer`.
 * On success, `out_plugin` owns the created layer and must be freed with
 * `layer_plugin_free`.
 * Returns 0 on success, -1 on failure.
 */
int layer_plugin_dense_create(int input_size,
                              int output_size,
                              Activation activation,
                              LayerPlugin *out_plugin);

/*
 * Frees resources owned by the plugin and resets function pointers.
 * Safe to call on partially initialized plugins.
 */
void layer_plugin_free(LayerPlugin *plugin);

#endif /* LAYERS_H */

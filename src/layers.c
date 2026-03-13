#include "../include/layers.h"

#include <stdlib.h>

static int dense_forward(void *ctx, const float *input, float *output) {
    return layer_forward((Layer *)ctx, input, output);
}

static int dense_backward(const void *ctx,
                         const float *delta_in,
                         float       *delta_out,
                         float       *grad_w,
                         float       *grad_b) {
    return layer_backward((const Layer *)ctx, delta_in, delta_out, grad_w, grad_b);
}

static int dense_input_size(const void *ctx) {
    return ((const Layer *)ctx)->input_size;
}

static int dense_output_size(const void *ctx) {
    return ((const Layer *)ctx)->output_size;
}

static float *dense_weights(void *ctx) {
    return ((Layer *)ctx)->weights;
}

static float *dense_biases(void *ctx) {
    return ((Layer *)ctx)->biases;
}

static int dense_weights_size(const void *ctx) {
    const Layer *layer = (const Layer *)ctx;
    return layer->input_size * layer->output_size;
}

static int dense_biases_size(const void *ctx) {
    return ((const Layer *)ctx)->output_size;
}

static void dense_destroy(void *ctx) {
    Layer *layer = (Layer *)ctx;
    if (!layer) return;
    layer_free(layer);
    free(layer);
}

int layer_plugin_dense_create(int input_size,
                              int output_size,
                              Activation activation,
                              LayerPlugin *out_plugin) {
    if (!out_plugin || input_size <= 0 || output_size <= 0) return -1;

    Layer *layer = malloc(sizeof(Layer));
    if (!layer) return -1;

    if (layer_init(layer, input_size, output_size, activation) != 0) {
        free(layer);
        return -1;
    }

    out_plugin->ctx = layer;
    out_plugin->forward = dense_forward;
    out_plugin->backward = dense_backward;
    out_plugin->input_size = dense_input_size;
    out_plugin->output_size = dense_output_size;
    out_plugin->weights = dense_weights;
    out_plugin->biases = dense_biases;
    out_plugin->weights_size = dense_weights_size;
    out_plugin->biases_size = dense_biases_size;
    out_plugin->destroy = dense_destroy;

    return 0;
}

void layer_plugin_free(LayerPlugin *plugin) {
    if (!plugin) return;

    if (plugin->destroy) {
        plugin->destroy(plugin->ctx);
    }

    plugin->ctx = NULL;
    plugin->forward = NULL;
    plugin->backward = NULL;
    plugin->input_size = NULL;
    plugin->output_size = NULL;
    plugin->weights = NULL;
    plugin->biases = NULL;
    plugin->weights_size = NULL;
    plugin->biases_size = NULL;
    plugin->destroy = NULL;
}

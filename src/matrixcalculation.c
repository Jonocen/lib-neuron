#include "../include/matrixcalculation.h"
#include <stdlib.h>
#include <string.h>

/* Activation functions */

float act_apply(float x, Activation a) {
    switch (a) {
        case ACT_RELU:    return x > 0.0f ? x : 0.0f;
        case ACT_SIGMOID: return 1.0f / (1.0f + expf(-x));
        case ACT_TANH:    return tanhf(x);
        case ACT_LINEAR:  /* fall-through */
        default:          return x;
    }
}

float act_deriv(float x, Activation a) {
    float s;
    switch (a) {
        case ACT_RELU:    return x > 0.0f ? 1.0f : 0.0f;
        case ACT_SIGMOID: s = act_apply(x, ACT_SIGMOID); return s * (1.0f - s);
        case ACT_TANH:    s = tanhf(x); return 1.0f - s * s;
        case ACT_LINEAR:  /* fall-through */
        default:          return 1.0f;
    }
}
 
int layer_init(Layer *layer, int input_size, int output_size, Activation activation) {
    if (!layer || input_size <= 0 || output_size <= 0) return -1;
    layer->input_size  = input_size;
    layer->output_size = output_size;
    layer->activation  = activation;
    layer->weights     = calloc(output_size * input_size, sizeof(float));
    layer->biases      = calloc(output_size,              sizeof(float));
    layer->cache_input = calloc(input_size,               sizeof(float));
    layer->cache_z     = calloc(output_size,              sizeof(float));
    if (!layer->weights || !layer->biases || !layer->cache_input || !layer->cache_z) {
        layer_free(layer);
        return -1;
    }
    return 0;
}

void layer_free(Layer *layer) {
    if (!layer) return;
    free(layer->weights);     layer->weights     = NULL;
    free(layer->biases);      layer->biases      = NULL;
    free(layer->cache_input); layer->cache_input = NULL;
    free(layer->cache_z);     layer->cache_z     = NULL;
}

/* Forward pass */
int layer_forward(Layer *layer, const float *input, float *output) {
    if (!layer || !input || !output) return -1;
    int in  = layer->input_size;
    int out = layer->output_size;

    memcpy(layer->cache_input, input, in * sizeof(float));

    for (int i = 0; i < out; i++) {
        float z = layer->biases[i];
        for (int j = 0; j < in; j++)
            z += layer->weights[i * in + j] * input[j];
        layer->cache_z[i] = z;
        output[i]         = act_apply(z, layer->activation);
    }
    return 0;
}

/* Backpropagation */

int layer_backward(const Layer *layer,
                   const float *delta_in,
                   float       *delta_out,
                   float       *grad_w,
                   float       *grad_b) {
    if (!layer || !delta_in || !grad_w || !grad_b) return -1;
    int in  = layer->input_size;
    int out = layer->output_size;

    /* Reuse grad_b as the layer-local delta buffer to avoid heap allocation. */
    for (int i = 0; i < out; i++)
        grad_b[i] = delta_in[i] * act_deriv(layer->cache_z[i], layer->activation);

    /* weight and bias gradients */
    for (int i = 0; i < out; i++) {
        for (int j = 0; j < in; j++)
            grad_w[i * in + j] = grad_b[i] * layer->cache_input[j];
    }

    /* propagate error to previous layer: delta_out = W^T * delta */
    if (delta_out) {
        for (int j = 0; j < in; j++) {
            float sum = 0.0f;
            for (int i = 0; i < out; i++)
                sum += layer->weights[i * in + j] * grad_b[i];
            delta_out[j] = sum;
        }
    }

    return 0;
}

static int conv2d_input_index(const Conv2DLayer *layer, int c, int y, int x) {
    return (c * layer->input_height + y) * layer->input_width + x;
}

static int conv2d_output_index(const Conv2DLayer *layer, int c, int y, int x) {
    return (c * layer->output_height + y) * layer->output_width + x;
}

static int conv2d_weight_index(const Conv2DLayer *layer, int oc, int ic, int ky, int kx) {
    int kernel_size = layer->kernel_height * layer->kernel_width;
    int in_kernel_size = layer->input_channels * kernel_size;
    return oc * in_kernel_size + ic * kernel_size + ky * layer->kernel_width + kx;
}

int conv2d_layer_init(Conv2DLayer *layer,
                      int input_width,
                      int input_height,
                      int input_channels,
                      int output_channels,
                      int kernel_width,
                      int kernel_height,
                      int stride,
                      int padding,
                      Activation activation) {
    if (!layer || input_width <= 0 || input_height <= 0 || input_channels <= 0 ||
        output_channels <= 0 || kernel_width <= 0 || kernel_height <= 0 ||
        stride <= 0 || padding < 0) {
        return -1;
    }

    int padded_w = input_width + 2 * padding;
    int padded_h = input_height + 2 * padding;
    if (padded_w < kernel_width || padded_h < kernel_height) return -1;

    int out_w_numerator = padded_w - kernel_width;
    int out_h_numerator = padded_h - kernel_height;
    if ((out_w_numerator % stride) != 0 || (out_h_numerator % stride) != 0) return -1;

    int output_width = (out_w_numerator / stride) + 1;
    int output_height = (out_h_numerator / stride) + 1;
    if (output_width <= 0 || output_height <= 0) return -1;

    layer->input_width = input_width;
    layer->input_height = input_height;
    layer->input_channels = input_channels;
    layer->output_channels = output_channels;
    layer->kernel_width = kernel_width;
    layer->kernel_height = kernel_height;
    layer->stride = stride;
    layer->padding = padding;
    layer->output_width = output_width;
    layer->output_height = output_height;
    layer->activation = activation;

    int input_size = input_width * input_height * input_channels;
    int output_size = output_width * output_height * output_channels;
    int weights_size = output_channels * input_channels * kernel_width * kernel_height;

    layer->weights = calloc((size_t)weights_size, sizeof(float));
    layer->biases = calloc((size_t)output_channels, sizeof(float));
    layer->cache_input = calloc((size_t)input_size, sizeof(float));
    layer->cache_z = calloc((size_t)output_size, sizeof(float));
    if (!layer->weights || !layer->biases || !layer->cache_input || !layer->cache_z) {
        conv2d_layer_free(layer);
        return -1;
    }

    return 0;
}

void conv2d_layer_free(Conv2DLayer *layer) {
    if (!layer) return;
    free(layer->weights);
    layer->weights = NULL;
    free(layer->biases);
    layer->biases = NULL;
    free(layer->cache_input);
    layer->cache_input = NULL;
    free(layer->cache_z);
    layer->cache_z = NULL;
}

int conv2d_layer_forward(Conv2DLayer *layer, const float *input, float *output) {
    if (!layer || !input || !output) return -1;

    int input_size = layer->input_width * layer->input_height * layer->input_channels;
    memcpy(layer->cache_input, input, (size_t)input_size * sizeof(float));

    for (int oc = 0; oc < layer->output_channels; oc++) {
        for (int oy = 0; oy < layer->output_height; oy++) {
            for (int ox = 0; ox < layer->output_width; ox++) {
                float z = layer->biases[oc];
                int in_y_origin = oy * layer->stride - layer->padding;
                int in_x_origin = ox * layer->stride - layer->padding;

                for (int ic = 0; ic < layer->input_channels; ic++) {
                    for (int ky = 0; ky < layer->kernel_height; ky++) {
                        int in_y = in_y_origin + ky;
                        if (in_y < 0 || in_y >= layer->input_height) continue;

                        for (int kx = 0; kx < layer->kernel_width; kx++) {
                            int in_x = in_x_origin + kx;
                            if (in_x < 0 || in_x >= layer->input_width) continue;

                            int input_idx = conv2d_input_index(layer, ic, in_y, in_x);
                            int weight_idx = conv2d_weight_index(layer, oc, ic, ky, kx);
                            z += layer->weights[weight_idx] * input[input_idx];
                        }
                    }
                }

                int output_idx = conv2d_output_index(layer, oc, oy, ox);
                layer->cache_z[output_idx] = z;
                output[output_idx] = act_apply(z, layer->activation);
            }
        }
    }

    return 0;
}

int conv2d_layer_backward(const Conv2DLayer *layer,
                          const float *delta_in,
                          float       *delta_out,
                          float       *grad_w,
                          float       *grad_b) {
    if (!layer || !delta_in || !grad_w || !grad_b) return -1;

    int input_size = layer->input_width * layer->input_height * layer->input_channels;
    int weights_size = layer->output_channels * layer->input_channels *
                       layer->kernel_width * layer->kernel_height;

    memset(grad_w, 0, (size_t)weights_size * sizeof(float));
    memset(grad_b, 0, (size_t)layer->output_channels * sizeof(float));
    if (delta_out) {
        memset(delta_out, 0, (size_t)input_size * sizeof(float));
    }

    for (int oc = 0; oc < layer->output_channels; oc++) {
        for (int oy = 0; oy < layer->output_height; oy++) {
            for (int ox = 0; ox < layer->output_width; ox++) {
                int output_idx = conv2d_output_index(layer, oc, oy, ox);
                float d = delta_in[output_idx] * act_deriv(layer->cache_z[output_idx], layer->activation);
                grad_b[oc] += d;

                int in_y_origin = oy * layer->stride - layer->padding;
                int in_x_origin = ox * layer->stride - layer->padding;

                for (int ic = 0; ic < layer->input_channels; ic++) {
                    for (int ky = 0; ky < layer->kernel_height; ky++) {
                        int in_y = in_y_origin + ky;
                        if (in_y < 0 || in_y >= layer->input_height) continue;

                        for (int kx = 0; kx < layer->kernel_width; kx++) {
                            int in_x = in_x_origin + kx;
                            if (in_x < 0 || in_x >= layer->input_width) continue;

                            int input_idx = conv2d_input_index(layer, ic, in_y, in_x);
                            int weight_idx = conv2d_weight_index(layer, oc, ic, ky, kx);
                            grad_w[weight_idx] += d * layer->cache_input[input_idx];
                            if (delta_out) {
                                delta_out[input_idx] += layer->weights[weight_idx] * d;
                            }
                        }
                    }
                }
            }
        }
    }

    return 0;
}

static int maxpool2d_input_index(const MaxPool2DLayer *layer, int c, int y, int x) {
    return (c * layer->input_height + y) * layer->input_width + x;
}

static int maxpool2d_output_index(const MaxPool2DLayer *layer, int c, int y, int x) {
    return (c * layer->output_height + y) * layer->output_width + x;
}

int maxpool2d_layer_init(MaxPool2DLayer *layer,
                         int input_width,
                         int input_height,
                         int channels,
                         int pool_width,
                         int pool_height,
                         int stride,
                         int padding) {
    if (!layer || input_width <= 0 || input_height <= 0 || channels <= 0 ||
        pool_width <= 0 || pool_height <= 0 || stride <= 0 || padding < 0) {
        return -1;
    }

    int padded_w = input_width + 2 * padding;
    int padded_h = input_height + 2 * padding;
    if (padded_w < pool_width || padded_h < pool_height) return -1;

    int out_w_numerator = padded_w - pool_width;
    int out_h_numerator = padded_h - pool_height;
    if ((out_w_numerator % stride) != 0 || (out_h_numerator % stride) != 0) return -1;

    int output_width = (out_w_numerator / stride) + 1;
    int output_height = (out_h_numerator / stride) + 1;
    if (output_width <= 0 || output_height <= 0) return -1;

    layer->input_width = input_width;
    layer->input_height = input_height;
    layer->channels = channels;
    layer->pool_width = pool_width;
    layer->pool_height = pool_height;
    layer->stride = stride;
    layer->padding = padding;
    layer->output_width = output_width;
    layer->output_height = output_height;

    int input_size = input_width * input_height * channels;
    int output_size = output_width * output_height * channels;

    layer->cache_input = calloc((size_t)input_size, sizeof(float));
    layer->cache_max_indices = malloc((size_t)output_size * sizeof(int));
    layer->dummy_weight = calloc(1, sizeof(float));
    layer->dummy_bias = calloc(1, sizeof(float));

    if (!layer->cache_input || !layer->cache_max_indices || !layer->dummy_weight || !layer->dummy_bias) {
        maxpool2d_layer_free(layer);
        return -1;
    }

    return 0;
}

void maxpool2d_layer_free(MaxPool2DLayer *layer) {
    if (!layer) return;
    free(layer->cache_input);
    layer->cache_input = NULL;
    free(layer->cache_max_indices);
    layer->cache_max_indices = NULL;
    free(layer->dummy_weight);
    layer->dummy_weight = NULL;
    free(layer->dummy_bias);
    layer->dummy_bias = NULL;
}

int maxpool2d_layer_forward(MaxPool2DLayer *layer, const float *input, float *output) {
    if (!layer || !input || !output) return -1;

    int input_size = layer->input_width * layer->input_height * layer->channels;
    memcpy(layer->cache_input, input, (size_t)input_size * sizeof(float));

    for (int c = 0; c < layer->channels; c++) {
        for (int oy = 0; oy < layer->output_height; oy++) {
            for (int ox = 0; ox < layer->output_width; ox++) {
                int in_y_origin = oy * layer->stride - layer->padding;
                int in_x_origin = ox * layer->stride - layer->padding;
                float max_value = -INFINITY;
                int max_input_index = -1;

                for (int py = 0; py < layer->pool_height; py++) {
                    int in_y = in_y_origin + py;
                    if (in_y < 0 || in_y >= layer->input_height) continue;

                    for (int px = 0; px < layer->pool_width; px++) {
                        int in_x = in_x_origin + px;
                        if (in_x < 0 || in_x >= layer->input_width) continue;

                        int input_idx = maxpool2d_input_index(layer, c, in_y, in_x);
                        float value = input[input_idx];
                        if (value > max_value) {
                            max_value = value;
                            max_input_index = input_idx;
                        }
                    }
                }

                if (max_input_index < 0) {
                    max_value = 0.0f;
                }

                int output_idx = maxpool2d_output_index(layer, c, oy, ox);
                output[output_idx] = max_value;
                layer->cache_max_indices[output_idx] = max_input_index;
            }
        }
    }

    return 0;
}

int maxpool2d_layer_backward(const MaxPool2DLayer *layer,
                             const float *delta_in,
                             float       *delta_out,
                             float       *grad_w,
                             float       *grad_b) {
    if (!layer || !delta_in || !grad_w || !grad_b) return -1;

    int output_size = layer->output_width * layer->output_height * layer->channels;
    int input_size = layer->input_width * layer->input_height * layer->channels;

    grad_w[0] = 0.0f;
    grad_b[0] = 0.0f;

    if (delta_out) {
        memset(delta_out, 0, (size_t)input_size * sizeof(float));
    }

    for (int i = 0; i < output_size; i++) {
        int input_idx = layer->cache_max_indices[i];
        if (input_idx >= 0 && delta_out) {
            delta_out[input_idx] += delta_in[i];
        }
    }

    return 0;
}
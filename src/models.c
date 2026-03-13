#include "../include/models.h"

#include <stdlib.h>
#include <string.h>

static int plugin_layer_valid(const LayerPlugin *layer) {
    if (!layer) return 0;
    return layer->ctx &&
           layer->forward &&
           layer->backward &&
           layer->input_size &&
           layer->output_size &&
           layer->weights &&
           layer->biases &&
           layer->weights_size &&
           layer->biases_size &&
           layer->destroy;
}

static int max_plugin_layer_width(const SequentialModel *model) {
    int max_width = 0;

    for (int i = 0; i < model->num_layers; i++) {
        int in = model->layers[i].input_size(model->layers[i].ctx);
        int out = model->layers[i].output_size(model->layers[i].ctx);

        if (in > max_width) {
            max_width = in;
        }
        if (out > max_width) {
            max_width = out;
        }
    }

    return max_width;
}

int sequential_model_init(SequentialModel *model, int initial_capacity) {
    if (!model || initial_capacity <= 0) return -1;

    model->layers = calloc((size_t)initial_capacity, sizeof(LayerPlugin));
    if (!model->layers) return -1;

    model->num_layers = 0;
    model->capacity = initial_capacity;
    return 0;
}

void sequential_model_free(SequentialModel *model) {
    if (!model) return;

    for (int i = 0; i < model->num_layers; i++) {
        layer_plugin_free(&model->layers[i]);
    }

    free(model->layers);
    model->layers = NULL;
    model->num_layers = 0;
    model->capacity = 0;
}

int sequential_model_add_layer(SequentialModel *model, LayerPlugin layer) {
    if (!model || !plugin_layer_valid(&layer)) return -1;

    if (model->num_layers >= model->capacity) {
        int new_capacity = model->capacity * 2;
        LayerPlugin *new_layers = realloc(model->layers, (size_t)new_capacity * sizeof(LayerPlugin));
        if (!new_layers) {
            return -1;
        }

        model->layers = new_layers;
        model->capacity = new_capacity;
    }

    model->layers[model->num_layers++] = layer;
    return 0;
}

int sequential_model_add_dense(SequentialModel *model,
                               int input_size,
                               int output_size,
                               Activation activation) {
    LayerPlugin layer;
    memset(&layer, 0, sizeof(LayerPlugin));

    if (!model) return -1;

    if (layer_plugin_dense_create(input_size, output_size, activation, &layer) != 0) {
        return -1;
    }

    if (sequential_model_add_layer(model, layer) != 0) {
        layer_plugin_free(&layer);
        return -1;
    }

    return 0;
}

int sequential_model_forward(SequentialModel *model,
                             const float *input,
                             float *output) {
    if (!model || model->num_layers <= 0 || !input || !output) return -1;

    if (model->num_layers == 1) {
        return model->layers[0].forward(model->layers[0].ctx, input, output);
    }

    int width = max_plugin_layer_width(model);
    float *buffer_a = malloc((size_t)width * sizeof(float));
    float *buffer_b = malloc((size_t)width * sizeof(float));
    if (!buffer_a || !buffer_b) {
        free(buffer_a);
        free(buffer_b);
        return -1;
    }

    const float *current_input = input;
    for (int i = 0; i < model->num_layers; i++) {
        int is_last = (i == model->num_layers - 1);
        float *current_output = is_last ? output : ((i % 2 == 0) ? buffer_a : buffer_b);

        if (model->layers[i].forward(model->layers[i].ctx, current_input, current_output) != 0) {
            free(buffer_a);
            free(buffer_b);
            return -1;
        }

        current_input = current_output;
    }

    free(buffer_a);
    free(buffer_b);
    return 0;
}

int sequential_model_train_step_sgd(SequentialModel *model,
                                    const float *input,
                                    const float *target,
                                    float *output,
                                    float learning_rate,
                                    float *loss_out) {
    if (!model || model->num_layers <= 0 || !input || !target || !output) {
        return -1;
    }

    if (sequential_model_forward(model, input, output) != 0) {
        return -1;
    }

    int output_size = model->layers[model->num_layers - 1].output_size(
        model->layers[model->num_layers - 1].ctx);
    if (loss_out) {
        *loss_out = loss_mse(output, target, output_size);
    }

    int width = max_plugin_layer_width(model);
    float *delta_curr = malloc((size_t)width * sizeof(float));
    float *delta_prev = malloc((size_t)width * sizeof(float));
    if (!delta_curr || !delta_prev) {
        free(delta_curr);
        free(delta_prev);
        return -1;
    }

    if (loss_mse_grad(output, target, output_size, delta_curr) != 0) {
        free(delta_curr);
        free(delta_prev);
        return -1;
    }

    for (int i = model->num_layers - 1; i >= 0; i--) {
        float *next_delta = (i > 0) ? delta_prev : NULL;

        int grad_w_size = model->layers[i].weights_size(model->layers[i].ctx);
        int grad_b_size = model->layers[i].biases_size(model->layers[i].ctx);

        float *grad_w = malloc((size_t)grad_w_size * sizeof(float));
        float *grad_b = malloc((size_t)grad_b_size * sizeof(float));
        if (!grad_w || !grad_b) {
            free(grad_w);
            free(grad_b);
            free(delta_curr);
            free(delta_prev);
            return -1;
        }

        if (model->layers[i].backward(model->layers[i].ctx, delta_curr, next_delta, grad_w, grad_b) != 0) {
            free(grad_w);
            free(grad_b);
            free(delta_curr);
            free(delta_prev);
            return -1;
        }

        if (sgd_optimizer(model->layers[i].weights(model->layers[i].ctx), grad_w, learning_rate, grad_w_size) != 0 ||
            sgd_optimizer(model->layers[i].biases(model->layers[i].ctx), grad_b, learning_rate, grad_b_size) != 0) {
            free(grad_w);
            free(grad_b);
            free(delta_curr);
            free(delta_prev);
            return -1;
        }

        free(grad_w);
        free(grad_b);

        if (i > 0) {
            float *tmp = delta_curr;
            delta_curr = delta_prev;
            delta_prev = tmp;
        }
    }

    free(delta_curr);
    free(delta_prev);
    return 0;
}

static int max_layer_width(const Layer *layers, int num_layers) {
    int max_width = 0;

    for (int i = 0; i < num_layers; i++) {
        if (layers[i].input_size > max_width) {
            max_width = layers[i].input_size;
        }
        if (layers[i].output_size > max_width) {
            max_width = layers[i].output_size;
        }
    }

    return max_width;
}

int sequential_forward(Layer *layers, int num_layers, const float *input, float *output) {
    if (!layers || num_layers <= 0 || !input || !output) return -1;

    if (num_layers == 1) {
        return layer_forward(&layers[0], input, output);
    }

    int width = max_layer_width(layers, num_layers);
    float *buffer_a = malloc((size_t)width * sizeof(float));
    float *buffer_b = malloc((size_t)width * sizeof(float));
    if (!buffer_a || !buffer_b) {
        free(buffer_a);
        free(buffer_b);
        return -1;
    }

    const float *current_input = input;
    for (int i = 0; i < num_layers; i++) {
        int is_last = (i == num_layers - 1);
        float *current_output = is_last ? output : ((i % 2 == 0) ? buffer_a : buffer_b);

        if (layer_forward(&layers[i], current_input, current_output) != 0) {
            free(buffer_a);
            free(buffer_b);
            return -1;
        }

        current_input = current_output;
    }

    free(buffer_a);
    free(buffer_b);
    return 0;
}

int sequential_train_step_sgd(Layer *layers, int num_layers,
                              const float *input, const float *target,
                              float *output,
                              float **grads_w, float **grads_b,
                              float learning_rate,
                              float *loss_out) {
    if (!layers || num_layers <= 0 || !input || !target || !output || !grads_w || !grads_b) {
        return -1;
    }

    if (sequential_forward(layers, num_layers, input, output) != 0) {
        return -1;
    }

    int output_size = layers[num_layers - 1].output_size;
    if (loss_out) {
        *loss_out = loss_mse(output, target, output_size);
    }

    int width = max_layer_width(layers, num_layers);
    float *delta_curr = malloc((size_t)width * sizeof(float));
    float *delta_prev = malloc((size_t)width * sizeof(float));
    if (!delta_curr || !delta_prev) {
        free(delta_curr);
        free(delta_prev);
        return -1;
    }

    if (loss_mse_grad(output, target, output_size, delta_curr) != 0) {
        free(delta_curr);
        free(delta_prev);
        return -1;
    }

    for (int i = num_layers - 1; i >= 0; i--) {
        float *next_delta = (i > 0) ? delta_prev : NULL;
        int grad_w_size = layers[i].output_size * layers[i].input_size;
        int grad_b_size = layers[i].output_size;

        if (!grads_w[i] || !grads_b[i]) {
            free(delta_curr);
            free(delta_prev);
            return -1;
        }

        if (layer_backward(&layers[i], delta_curr, next_delta, grads_w[i], grads_b[i]) != 0) {
            free(delta_curr);
            free(delta_prev);
            return -1;
        }

        if (sgd_optimizer(layers[i].weights, grads_w[i], learning_rate, grad_w_size) != 0 ||
            sgd_optimizer(layers[i].biases, grads_b[i], learning_rate, grad_b_size) != 0) {
            free(delta_curr);
            free(delta_prev);
            return -1;
        }

        if (i > 0) {
            float *tmp = delta_curr;
            delta_curr = delta_prev;
            delta_prev = tmp;
        }
    }

    free(delta_curr);
    free(delta_prev);
    return 0;
}
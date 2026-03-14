#include "../include/models.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LNN_MAGIC "LNN1"
#define LNN_MAGIC_SIZE 4

/* TODO LIST                        Done?

Add New optimizers                      NO
Add More activation functions           NO
Fix Performence                         No
Check For Flaws in code                 NO



*/





static int has_lnn_extension(const char *file_path) {
    if (!file_path) return 0;

    const char *dot = strrchr(file_path, '.');
    if (!dot) return 0;

    return strcmp(dot, ".lnn") == 0;
}

static void free_layer_snapshots(float **weights,
                                 float **biases,
                                 int num_layers) {
    if (weights) {
        for (int i = 0; i < num_layers; i++) {
            free(weights[i]);
        }
        free(weights);
    }

    if (biases) {
        for (int i = 0; i < num_layers; i++) {
            free(biases[i]);
        }
        free(biases);
    }
}

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

static void model_workspace_free(SequentialModel *model) {
    if (!model) return;

    free(model->work_forward_a);
    free(model->work_forward_b);
    free(model->work_delta_a);
    free(model->work_delta_b);
    free(model->work_grad_w);
    free(model->work_grad_b);
    model->work_forward_a = NULL;
    model->work_forward_b = NULL;
    model->work_forward_size = 0;
    model->work_delta_a = NULL;
    model->work_delta_b = NULL;
    model->work_delta_size = 0;
    model->work_grad_w = NULL;
    model->work_grad_b = NULL;
    model->work_grad_w_size = 0;
    model->work_grad_b_size = 0;
}

static int ensure_workspace(float **buffer, int *current_size, int required_size) {
    if (!buffer || !current_size || required_size <= 0) return -1;

    if (*buffer && *current_size >= required_size) {
        return 0;
    }

    float *new_buffer = realloc(*buffer, (size_t)required_size * sizeof(float));
    if (!new_buffer) return -1;

    *buffer = new_buffer;
    *current_size = required_size;
    return 0;
}

static void free_grad_accumulators(float **acc_w,
                                   float **acc_b,
                                   int num_layers) {
    if (acc_w) {
        for (int i = 0; i < num_layers; i++) {
            free(acc_w[i]);
        }
        free(acc_w);
    }

    if (acc_b) {
        for (int i = 0; i < num_layers; i++) {
            free(acc_b[i]);
        }
        free(acc_b);
    }
}

static int adam_optimizer_state_valid(const OptimizerState *optimizer_state) {
    if (!optimizer_state) return 0;
    if (!optimizer_state->m_w || !optimizer_state->v_w || !optimizer_state->m_b || !optimizer_state->v_b) return 0;
    if (optimizer_state->step <= 0) return 0;
    if (optimizer_state->beta1 <= 0.0f || optimizer_state->beta1 >= 1.0f) return 0;
    if (optimizer_state->beta2 <= 0.0f || optimizer_state->beta2 >= 1.0f) return 0;
    return 1;
}

static int rmsprop_optimizer_state_valid(const OptimizerState *optimizer_state) {
    if (!optimizer_state) return 0;
    if (!optimizer_state->m_w || !optimizer_state->m_b) return 0;
    if (optimizer_state->beta1 <= 0.0f || optimizer_state->beta1 >= 1.0f) return 0;
    return 1;
}

static int adagrad_optimizer_state_valid(const OptimizerState *optimizer_state) {
    if (!optimizer_state) return 0;
    if (!optimizer_state->m_w || !optimizer_state->m_b) return 0;
    return 1;
}

static int optimizer_state_valid(OptimizerType optimizer,
                                 const OptimizerState *optimizer_state) {
    if (optimizer == OPTIMIZER_ADAM) {
        return adam_optimizer_state_valid(optimizer_state);
    }
    if (optimizer == OPTIMIZER_RMSPROP) {
        return rmsprop_optimizer_state_valid(optimizer_state);
    }
    if (optimizer == OPTIMIZER_ADAGRAD) {
        return adagrad_optimizer_state_valid(optimizer_state);
    }
    return 1;
}

//todo: add New Optimizers and their updates

static int apply_optimizer_update(float *weights,
                                  const float *grads,
                                  int size,
                                  OptimizerType optimizer,
                                  float learning_rate,
                                  OptimizerState *optimizer_state,
                                  float *opt_state_a,
                                  float *opt_state_b) {
    if (!weights || !grads || size <= 0) return -1;

    if (optimizer == OPTIMIZER_SGD) {
        return sgd_optimizer(weights, (float *)grads, learning_rate, size);
    }
    if (optimizer == OPTIMIZER_ADAM) {
        if (!optimizer_state || !opt_state_a || !opt_state_b) return -1;
        return adam_optimizer(weights,
                              (float *)grads,
                              opt_state_a,
                              opt_state_b,
                              optimizer_state->beta1,
                              optimizer_state->beta2,
                              learning_rate,
                              optimizer_state->step,
                              size);
    }

    if (optimizer == OPTIMIZER_RMSPROP) {
        if (!optimizer_state || !opt_state_a) return -1;
        return rmsprop_optimizer(weights,
                                 (float *)grads,
                                 opt_state_a,
                                 optimizer_state->beta1,
                                 learning_rate,
                                 size);
    }

    if (optimizer == OPTIMIZER_ADAGRAD) {
        if (!optimizer_state || !opt_state_a) return -1;
        return adagrad_optimizer(weights,
                                 (float *)grads,
                                 opt_state_a,
                                 learning_rate,
                                 size);
    }

    return -1;
}

static int compute_loss_and_grad(LossFunctionType loss_function,
                                 const float *prediction,
                                 const float *target,
                                 int size,
                                 float *loss_out,
                                 float *grad_out) {
    const float huber_delta = 1.0f;

    if (!prediction || !target || !grad_out || size <= 0) return -1;

    if (loss_function == LOSS_MSE) {
        if (loss_out) {
            *loss_out = loss_mse(prediction, target, size);
        }
        return loss_mse_grad(prediction, target, size, grad_out);
    }

    if (loss_function == LOSS_BCE) {
        if (loss_out) {
            *loss_out = loss_bce(prediction, target, size);
        }
        return loss_bce_grad(prediction, target, size, grad_out);
    }

    if (loss_function == LOSS_HUBER) {
        if (loss_out) {
            *loss_out = loss_huber(prediction, target, size, huber_delta);
        }
        return loss_huber_grad(prediction, target, size, huber_delta, grad_out);
    }

    return -1;
}

//Todo: Make a new init function

int sequential_model_init(SequentialModel *model, int initial_capacity) {
    if (!model || initial_capacity <= 0) return -1;

    model->layers = calloc((size_t)initial_capacity, sizeof(LayerPlugin));
    if (!model->layers) return -1;

    model->num_layers = 0;
    model->capacity = initial_capacity;
    model->compiled = 0;
    model->compiled_loss = LOSS_MSE;
    model->compiled_optimizer = OPTIMIZER_SGD;
    model->compiled_learning_rate = 0.0f;
    model->compiled_owns_optimizer_state = 0;
    model->compiled_optimizer_state = (OptimizerState){0};
    model->work_forward_a = NULL;
    model->work_forward_b = NULL;
    model->work_forward_size = 0;
    model->work_delta_a = NULL;
    model->work_delta_b = NULL;
    model->work_delta_size = 0;
    model->work_grad_w = NULL;
    model->work_grad_b = NULL;
    model->work_grad_w_size = 0;
    model->work_grad_b_size = 0;
    return 0;
}

//Todo: make it easyer for new Optimizers

void sequential_model_free(SequentialModel *model) {
    if (!model) return;

    if (model->compiled_owns_optimizer_state) {
        sequential_model_optimizer_state_free(model, &model->compiled_optimizer_state);
        model->compiled_owns_optimizer_state = 0;
    }

    model_workspace_free(model);

    for (int i = 0; i < model->num_layers; i++) {
        layer_plugin_free(&model->layers[i]);
    }

    free(model->layers);
    model->layers = NULL;
    model->num_layers = 0;
    model->capacity = 0;
    model->compiled = 0;
}

int sequential_model_add_layer(SequentialModel *model, LayerPlugin layer) {
    if (!model || !plugin_layer_valid(&layer)) return -1;

    if (model->compiled_owns_optimizer_state) {
        sequential_model_optimizer_state_free(model, &model->compiled_optimizer_state);
        model->compiled_owns_optimizer_state = 0;
    }

    //Todo: Rename APIs for better understandig + change variable names 
    model_workspace_free(model);
    model->compiled = 0;

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

int sequential_model_add_conv2d(SequentialModel *model,
                                int input_width,
                                int input_height,
                                int input_channels,
                                int output_channels,
                                int kernel_width,
                                int kernel_height,
                                int stride,
                                int padding,
                                Activation activation) {
    LayerPlugin layer;
    memset(&layer, 0, sizeof(LayerPlugin));

    if (!model) return -1;

    if (layer_plugin_conv2d_create(input_width,
                                   input_height,
                                   input_channels,
                                   output_channels,
                                   kernel_width,
                                   kernel_height,
                                   stride,
                                   padding,
                                   activation,
                                   &layer) != 0) {
        return -1;
    }

    if (sequential_model_add_layer(model, layer) != 0) {
        layer_plugin_free(&layer);
        return -1;
    }

    return 0;
}

int sequential_model_add_maxpool2d(SequentialModel *model,
                                   int input_width,
                                   int input_height,
                                   int channels,
                                   int pool_width,
                                   int pool_height,
                                   int stride,
                                   int padding) {
    LayerPlugin layer;
    memset(&layer, 0, sizeof(LayerPlugin));

    if (!model) return -1;

    if (layer_plugin_maxpool2d_create(input_width,
                                      input_height,
                                      channels,
                                      pool_width,
                                      pool_height,
                                      stride,
                                      padding,
                                      &layer) != 0) {
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

    if (!model->work_forward_a || !model->work_forward_b) {
        int width = max_plugin_layer_width(model);
        if (ensure_workspace(&model->work_forward_a, &model->work_forward_size, width) != 0 ||
            ensure_workspace(&model->work_forward_b, &model->work_forward_size, width) != 0) {
            return -1;
        }
    }

    const float *current_input = input;
    for (int i = 0; i < model->num_layers; i++) {
        int is_last = (i == model->num_layers - 1);
        float *current_output = is_last ? output : ((i % 2 == 0) ? model->work_forward_a : model->work_forward_b);

        if (model->layers[i].forward(model->layers[i].ctx, current_input, current_output) != 0) {
            return -1;
        }

        current_input = current_output;
    }

    return 0;
}

//Todo: Add seeding for rand

int sequential_model_randomize(SequentialModel *model, float init_scale) {
    if (!model || model->num_layers <= 0 || init_scale <= 0.0f) return -1;

    for (int l = 0; l < model->num_layers; l++) {
        float *w = model->layers[l].weights(model->layers[l].ctx);
        float *b = model->layers[l].biases(model->layers[l].ctx);
        int nw = model->layers[l].weights_size(model->layers[l].ctx);
        int nb = model->layers[l].biases_size(model->layers[l].ctx);

        if (!w || !b || nw <= 0 || nb <= 0) return -1;

        for (int i = 0; i < nw; i++) {
            w[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * init_scale;
        }
        for (int i = 0; i < nb; i++) {
            b[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * init_scale;
        }
    }

    return 0;
}

int sequential_model_predict(SequentialModel *model,
                             const float *input,
                             float *output) {
    return sequential_model_forward(model, input, output);
}

int sequential_model_save_lnn(const SequentialModel *model,
                              const char *file_path) {
    if (!model || !file_path || !model->layers || model->num_layers <= 0) return -1;
    if (!has_lnn_extension(file_path)) return -1;

    FILE *file = fopen(file_path, "wb");
    if (!file) return -1;

    uint32_t num_layers = (uint32_t)model->num_layers;

    if (fwrite(LNN_MAGIC, 1, LNN_MAGIC_SIZE, file) != LNN_MAGIC_SIZE ||
        fwrite(&num_layers, sizeof(num_layers), 1, file) != 1) {
        fclose(file);
        return -1;
    }

    for (int i = 0; i < model->num_layers; i++) {
        const LayerPlugin *layer = &model->layers[i];
        float *weights = layer->weights(layer->ctx);
        float *biases = layer->biases(layer->ctx);
        int weights_size = layer->weights_size(layer->ctx);
        int biases_size = layer->biases_size(layer->ctx);

        if (!weights || !biases || weights_size <= 0 || biases_size <= 0) {
            fclose(file);
            return -1;
        }

        uint32_t weights_size_u32 = (uint32_t)weights_size;
        uint32_t biases_size_u32 = (uint32_t)biases_size;

        if (fwrite(&weights_size_u32, sizeof(weights_size_u32), 1, file) != 1 ||
            fwrite(&biases_size_u32, sizeof(biases_size_u32), 1, file) != 1 ||
            fwrite(weights, sizeof(float), (size_t)weights_size, file) != (size_t)weights_size ||
            fwrite(biases, sizeof(float), (size_t)biases_size, file) != (size_t)biases_size) {
            fclose(file);
            return -1;
        }
    }

    if (fclose(file) != 0) return -1;
    return 0;
}

int sequential_model_load_lnn(SequentialModel *model,
                              const char *file_path) {
    if (!model || !file_path || !model->layers || model->num_layers <= 0) return -1;
    if (!has_lnn_extension(file_path)) return -1;

    FILE *file = fopen(file_path, "rb");
    if (!file) return -1;

    char magic[LNN_MAGIC_SIZE];
    uint32_t num_layers = 0;

    if (fread(magic, 1, LNN_MAGIC_SIZE, file) != LNN_MAGIC_SIZE ||
        fread(&num_layers, sizeof(num_layers), 1, file) != 1) {
        fclose(file);
        return -1;
    }

    if (memcmp(magic, LNN_MAGIC, LNN_MAGIC_SIZE) != 0 ||
        num_layers != (uint32_t)model->num_layers) {
        fclose(file);
        return -1;
    }

    float **weights_snapshots = calloc((size_t)model->num_layers, sizeof(float *));
    float **biases_snapshots = calloc((size_t)model->num_layers, sizeof(float *));
    if (!weights_snapshots || !biases_snapshots) {
        free_layer_snapshots(weights_snapshots, biases_snapshots, model->num_layers);
        fclose(file);
        return -1;
    }

    for (int i = 0; i < model->num_layers; i++) {
        LayerPlugin *layer = &model->layers[i];
        int expected_weights_size = layer->weights_size(layer->ctx);
        int expected_biases_size = layer->biases_size(layer->ctx);
        uint32_t file_weights_size = 0;
        uint32_t file_biases_size = 0;

        if (expected_weights_size <= 0 || expected_biases_size <= 0 ||
            fread(&file_weights_size, sizeof(file_weights_size), 1, file) != 1 ||
            fread(&file_biases_size, sizeof(file_biases_size), 1, file) != 1 ||
            file_weights_size != (uint32_t)expected_weights_size ||
            file_biases_size != (uint32_t)expected_biases_size) {
            free_layer_snapshots(weights_snapshots, biases_snapshots, model->num_layers);
            fclose(file);
            return -1;
        }

        weights_snapshots[i] = malloc((size_t)expected_weights_size * sizeof(float));
        biases_snapshots[i] = malloc((size_t)expected_biases_size * sizeof(float));
        if (!weights_snapshots[i] || !biases_snapshots[i]) {
            free_layer_snapshots(weights_snapshots, biases_snapshots, model->num_layers);
            fclose(file);
            return -1;
        }

        if (fread(weights_snapshots[i], sizeof(float), (size_t)expected_weights_size, file) != (size_t)expected_weights_size ||
            fread(biases_snapshots[i], sizeof(float), (size_t)expected_biases_size, file) != (size_t)expected_biases_size) {
            free_layer_snapshots(weights_snapshots, biases_snapshots, model->num_layers);
            fclose(file);
            return -1;
        }
    }

    // Reject files with trailing unexpected data.
    if (fgetc(file) != EOF) {
        free_layer_snapshots(weights_snapshots, biases_snapshots, model->num_layers);
        fclose(file);
        return -1;
    }

    for (int i = 0; i < model->num_layers; i++) {
        LayerPlugin *layer = &model->layers[i];
        float *weights = layer->weights(layer->ctx);
        float *biases = layer->biases(layer->ctx);
        int weights_size = layer->weights_size(layer->ctx);
        int biases_size = layer->biases_size(layer->ctx);

        if (!weights || !biases || weights_size <= 0 || biases_size <= 0) {
            free_layer_snapshots(weights_snapshots, biases_snapshots, model->num_layers);
            fclose(file);
            return -1;
        }

        memcpy(weights, weights_snapshots[i], (size_t)weights_size * sizeof(float));
        memcpy(biases, biases_snapshots[i], (size_t)biases_size * sizeof(float));
    }

    free_layer_snapshots(weights_snapshots, biases_snapshots, model->num_layers);

    if (fclose(file) != 0) return -1;
    return 0;
}

//Todo: renew the model_compile function and its variable names

int sequential_model_compile(SequentialModel *model,
                             LossFunctionType loss_function,
                             OptimizerType optimizer,
                             float learning_rate,
                             float optimizer_beta1,
                             float optimizer_beta2) {
    return sequential_model_compile_optimizer(model,
                                              loss_function,
                                              optimizer,
                                              learning_rate,
                                              optimizer_beta1,
                                              optimizer_beta2);
}

int sequential_model_compile_optimizer(SequentialModel *model,
                                       LossFunctionType loss_function,
                                       OptimizerType optimizer,
                                       float learning_rate,
                                       float optimizer_beta1,
                                       float optimizer_beta2) {
    if (!model || model->num_layers <= 0 || learning_rate <= 0.0f) return -1;
    if (loss_function != LOSS_MSE && loss_function != LOSS_BCE && loss_function != LOSS_HUBER) return -1;
    if (optimizer != OPTIMIZER_SGD && optimizer != OPTIMIZER_ADAM && optimizer != OPTIMIZER_RMSPROP && optimizer != OPTIMIZER_ADAGRAD) return -1;
    if (optimizer == OPTIMIZER_RMSPROP && learning_rate <= 0.0f) return -1;

    if (model->compiled_owns_optimizer_state) {
        sequential_model_optimizer_state_free(model, &model->compiled_optimizer_state);
        model->compiled_owns_optimizer_state = 0;
    }

    model->compiled = 0;
    model->compiled_loss = loss_function;
    model->compiled_optimizer = optimizer;
    model->compiled_learning_rate = learning_rate;

    if (optimizer == OPTIMIZER_ADAM || optimizer == OPTIMIZER_RMSPROP || optimizer == OPTIMIZER_ADAGRAD) {
        if (sequential_model_optimizer_state_init(model,
                                                  &model->compiled_optimizer_state,
                                                  optimizer,
                                                  optimizer_beta1,
                                                  optimizer_beta2) != 0) {
            return -1;
        }
        model->compiled_owns_optimizer_state = 1;
    }

    /* Pre-allocate all runtime workspaces now that topology is fixed, so that
     * sequential_model_forward and sequential_model_optimize_from_prediction
     * can skip per-call layer scans entirely. */
    {
        int width = 0, max_grad_w_size = 0, max_grad_b_size = 0;
        for (int i = 0; i < model->num_layers; i++) {
            int in  = model->layers[i].input_size(model->layers[i].ctx);
            int out = model->layers[i].output_size(model->layers[i].ctx);
            int ws  = model->layers[i].weights_size(model->layers[i].ctx);
            int bs  = model->layers[i].biases_size(model->layers[i].ctx);
            if (in  > width)            width            = in;
            if (out > width)            width            = out;
            if (ws  > max_grad_w_size)  max_grad_w_size  = ws;
            if (bs  > max_grad_b_size)  max_grad_b_size  = bs;
        }
        if (width > 0 && max_grad_w_size > 0 && max_grad_b_size > 0) {
            if (ensure_workspace(&model->work_forward_a, &model->work_forward_size,  width) != 0 ||
                ensure_workspace(&model->work_forward_b, &model->work_forward_size,  width) != 0 ||
                ensure_workspace(&model->work_delta_a,   &model->work_delta_size,    width) != 0 ||
                ensure_workspace(&model->work_delta_b,   &model->work_delta_size,    width) != 0 ||
                ensure_workspace(&model->work_grad_w,    &model->work_grad_w_size,   max_grad_w_size) != 0 ||
                ensure_workspace(&model->work_grad_b,    &model->work_grad_b_size,   max_grad_b_size) != 0) {
                if (model->compiled_owns_optimizer_state) {
                    sequential_model_optimizer_state_free(model, &model->compiled_optimizer_state);
                    model->compiled_owns_optimizer_state = 0;
                }
                return -1;
            }
        }
    }

    model->compiled = 1;
    return 0;
}

int sequential_model_train_with_progress(SequentialModel *model,
                                         const float *inputs,
                                         const float *targets,
                                         int num_samples,
                                         int input_size,
                                         int target_size,
                                         int epochs,
                                         int batch_size,
                                         int progress_percent,
                                         float *final_loss_out) {
    int status = -1;
    float epoch_loss = 0.0f;
    float *output = NULL;
    float **acc_w = NULL;
    float **acc_b = NULL;
    float **layer_weights = NULL;
    float **layer_biases = NULL;
    int *grad_w_sizes = NULL;
    int *grad_b_sizes = NULL;

    if (!model || !inputs || !targets || num_samples <= 0 || input_size <= 0 || target_size <= 0 || epochs <= 0) {
        return -1;
    }
    if (batch_size <= 0) return -1;
    if (progress_percent > 100) progress_percent = 100;
    if (!model->compiled) return -1;
    if (model->num_layers <= 0) return -1;

    int expected_input_size = model->layers[0].input_size(model->layers[0].ctx);
    int expected_target_size = model->layers[model->num_layers - 1].output_size(model->layers[model->num_layers - 1].ctx);
    if (input_size != expected_input_size || target_size != expected_target_size) return -1;

    if (batch_size > num_samples) {
        batch_size = num_samples;
    }

    OptimizerState *optimizer_state =
        (model->compiled_optimizer == OPTIMIZER_ADAM ||
         model->compiled_optimizer == OPTIMIZER_RMSPROP ||
         model->compiled_optimizer == OPTIMIZER_ADAGRAD)
            ? &model->compiled_optimizer_state
            : NULL;

    if (!optimizer_state_valid(model->compiled_optimizer, optimizer_state)) {
        return -1;
    }

    output = malloc((size_t)expected_target_size * sizeof(float));
    if (!output) goto cleanup;

    layer_weights = calloc((size_t)model->num_layers, sizeof(float *));
    layer_biases = calloc((size_t)model->num_layers, sizeof(float *));
    grad_w_sizes = calloc((size_t)model->num_layers, sizeof(int));
    grad_b_sizes = calloc((size_t)model->num_layers, sizeof(int));
    if (!layer_weights || !layer_biases || !grad_w_sizes || !grad_b_sizes) {
        goto cleanup;
    }

    int max_grad_w_size = 0;
    int max_grad_b_size = 0;
    for (int i = 0; i < model->num_layers; i++) {
        layer_weights[i] = model->layers[i].weights(model->layers[i].ctx);
        layer_biases[i] = model->layers[i].biases(model->layers[i].ctx);
        grad_w_sizes[i] = model->layers[i].weights_size(model->layers[i].ctx);
        grad_b_sizes[i] = model->layers[i].biases_size(model->layers[i].ctx);

        if (!layer_weights[i] || !layer_biases[i] || grad_w_sizes[i] <= 0 || grad_b_sizes[i] <= 0) {
            goto cleanup;
        }

        if (grad_w_sizes[i] > max_grad_w_size) {
            max_grad_w_size = grad_w_sizes[i];
        }
        if (grad_b_sizes[i] > max_grad_b_size) {
            max_grad_b_size = grad_b_sizes[i];
        }
    }

    int width = max_plugin_layer_width(model);
    if (width <= 0 || max_grad_w_size <= 0 || max_grad_b_size <= 0 ||
        ensure_workspace(&model->work_delta_a, &model->work_delta_size, width) != 0 ||
        ensure_workspace(&model->work_delta_b, &model->work_delta_size, width) != 0 ||
        ensure_workspace(&model->work_grad_w, &model->work_grad_w_size, max_grad_w_size) != 0 ||
        ensure_workspace(&model->work_grad_b, &model->work_grad_b_size, max_grad_b_size) != 0) {
        goto cleanup;
    }

    int next_progress = progress_percent;

    if (batch_size == 1) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            epoch_loss = 0.0f;
            for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
                const float *sample_input = inputs + ((size_t)sample_idx * (size_t)input_size);
                const float *sample_target = targets + ((size_t)sample_idx * (size_t)target_size);
                float sample_loss = 0.0f;
                float *delta_curr = model->work_delta_a;
                float *delta_prev = model->work_delta_b;

                if (sequential_model_forward(model, sample_input, output) != 0 ||
                    compute_loss_and_grad(model->compiled_loss,
                                          output,
                                          sample_target,
                                          target_size,
                                          &sample_loss,
                                          delta_curr) != 0) {
                    goto cleanup;
                }

                epoch_loss += sample_loss;

                for (int l = model->num_layers - 1; l >= 0; l--) {
                    float *next_delta = (l > 0) ? delta_prev : NULL;
                    float *opt_state_w_a = NULL;
                    float *opt_state_w_b = NULL;
                    float *opt_state_b_a = NULL;
                    float *opt_state_b_b = NULL;

                    if (model->layers[l].backward(model->layers[l].ctx,
                                                  delta_curr,
                                                  next_delta,
                                                  model->work_grad_w,
                                                  model->work_grad_b) != 0) {
                        goto cleanup;
                    }

                    if (model->compiled_optimizer == OPTIMIZER_ADAM) {
                        opt_state_w_a = optimizer_state->m_w[l];
                        opt_state_w_b = optimizer_state->v_w[l];
                        opt_state_b_a = optimizer_state->m_b[l];
                        opt_state_b_b = optimizer_state->v_b[l];
                    } else if (model->compiled_optimizer == OPTIMIZER_RMSPROP) {
                        opt_state_w_a = optimizer_state->m_w[l];
                        opt_state_b_a = optimizer_state->m_b[l];
                    } else if (model->compiled_optimizer == OPTIMIZER_ADAGRAD) {
                        opt_state_w_a = optimizer_state->m_w[l];
                        opt_state_b_a = optimizer_state->m_b[l];
                    }

                    if (apply_optimizer_update(layer_weights[l],
                                               model->work_grad_w,
                                               grad_w_sizes[l],
                                               model->compiled_optimizer,
                                               model->compiled_learning_rate,
                                               optimizer_state,
                                               opt_state_w_a,
                                               opt_state_w_b) != 0 ||
                        apply_optimizer_update(layer_biases[l],
                                               model->work_grad_b,
                                               grad_b_sizes[l],
                                               model->compiled_optimizer,
                                               model->compiled_learning_rate,
                                               optimizer_state,
                                               opt_state_b_a,
                                               opt_state_b_b) != 0) {
                        goto cleanup;
                    }

                    if (l > 0) {
                        float *tmp = delta_curr;
                        delta_curr = delta_prev;
                        delta_prev = tmp;
                    }
                }

                if (model->compiled_optimizer == OPTIMIZER_ADAM) {
                    optimizer_state->step += 1;
                }
            }

            if (progress_percent > 0) {
                float avg_loss = epoch_loss / (float)num_samples;
                int current_percent = ((epoch + 1) * 100) / epochs;
                int is_final_epoch = (epoch + 1) == epochs;

                if (current_percent >= next_progress || is_final_epoch) {
                    printf("[train] %d%% epoch=%d/%d loss=%.6f\n",
                           current_percent,
                           epoch + 1,
                           epochs,
                           avg_loss);

                    while (next_progress <= current_percent && next_progress > 0) {
                        next_progress += progress_percent;
                    }
                }
            }
        }

        if (final_loss_out) {
            *final_loss_out = epoch_loss / (float)num_samples;
        }

        status = 0;
        goto cleanup;
    }

    acc_w = calloc((size_t)model->num_layers, sizeof(float *));
    acc_b = calloc((size_t)model->num_layers, sizeof(float *));
    if (!acc_w || !acc_b) {
        goto cleanup;
    }

    for (int i = 0; i < model->num_layers; i++) {
        acc_w[i] = calloc((size_t)grad_w_sizes[i], sizeof(float));
        acc_b[i] = calloc((size_t)grad_b_sizes[i], sizeof(float));
        if (!acc_w[i] || !acc_b[i]) {
            goto cleanup;
        }
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        epoch_loss = 0.0f;
        for (int batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
            int batch_end = batch_start + batch_size;
            if (batch_end > num_samples) batch_end = num_samples;
            int current_batch = batch_end - batch_start;

            for (int l = 0; l < model->num_layers; l++) {
                memset(acc_w[l], 0, (size_t)grad_w_sizes[l] * sizeof(float));
                memset(acc_b[l], 0, (size_t)grad_b_sizes[l] * sizeof(float));
            }

            for (int sample_idx = batch_start; sample_idx < batch_end; sample_idx++) {
                const float *sample_input = inputs + ((size_t)sample_idx * (size_t)input_size);
                const float *sample_target = targets + ((size_t)sample_idx * (size_t)target_size);
                float sample_loss = 0.0f;
                float *delta_curr = model->work_delta_a;
                float *delta_prev = model->work_delta_b;

                if (sequential_model_forward(model, sample_input, output) != 0 ||
                    compute_loss_and_grad(model->compiled_loss,
                                          output,
                                          sample_target,
                                          target_size,
                                          &sample_loss,
                                          delta_curr) != 0) {
                    goto cleanup;
                }

                epoch_loss += sample_loss;

                for (int l = model->num_layers - 1; l >= 0; l--) {
                    float *next_delta = (l > 0) ? delta_prev : NULL;

                    if (grad_w_sizes[l] > model->work_grad_w_size ||
                        grad_b_sizes[l] > model->work_grad_b_size ||
                        model->layers[l].backward(model->layers[l].ctx,
                                                  delta_curr,
                                                  next_delta,
                                                  model->work_grad_w,
                                                  model->work_grad_b) != 0) {
                        goto cleanup;
                    }

                    for (int k = 0; k < grad_w_sizes[l]; k++) {
                        acc_w[l][k] += model->work_grad_w[k];
                    }
                    for (int k = 0; k < grad_b_sizes[l]; k++) {
                        acc_b[l][k] += model->work_grad_b[k];
                    }

                    if (l > 0) {
                        float *tmp = delta_curr;
                        delta_curr = delta_prev;
                        delta_prev = tmp;
                    }
                }
            }

            float inv_batch = 1.0f / (float)current_batch;
            for (int l = 0; l < model->num_layers; l++) {
                float *opt_state_w_a = NULL;
                float *opt_state_w_b = NULL;
                float *opt_state_b_a = NULL;
                float *opt_state_b_b = NULL;

                for (int k = 0; k < grad_w_sizes[l]; k++) {
                    acc_w[l][k] *= inv_batch;
                }
                for (int k = 0; k < grad_b_sizes[l]; k++) {
                    acc_b[l][k] *= inv_batch;
                }

                if (model->compiled_optimizer == OPTIMIZER_ADAM) {
                    opt_state_w_a = optimizer_state->m_w[l];
                    opt_state_w_b = optimizer_state->v_w[l];
                    opt_state_b_a = optimizer_state->m_b[l];
                    opt_state_b_b = optimizer_state->v_b[l];
                } else if (model->compiled_optimizer == OPTIMIZER_RMSPROP) {
                    opt_state_w_a = optimizer_state->m_w[l];
                    opt_state_b_a = optimizer_state->m_b[l];
                } else if (model->compiled_optimizer == OPTIMIZER_ADAGRAD) {
                    opt_state_w_a = optimizer_state->m_w[l];
                    opt_state_b_a = optimizer_state->m_b[l];
                }

                if (apply_optimizer_update(layer_weights[l],
                                           acc_w[l],
                                           grad_w_sizes[l],
                                           model->compiled_optimizer,
                                           model->compiled_learning_rate,
                                           optimizer_state,
                                           opt_state_w_a,
                                           opt_state_w_b) != 0 ||
                    apply_optimizer_update(layer_biases[l],
                                           acc_b[l],
                                           grad_b_sizes[l],
                                           model->compiled_optimizer,
                                           model->compiled_learning_rate,
                                           optimizer_state,
                                           opt_state_b_a,
                                           opt_state_b_b) != 0) {
                    goto cleanup;
                }
            }

            if (model->compiled_optimizer == OPTIMIZER_ADAM) {
                optimizer_state->step += 1;
            }
        }

        if (progress_percent > 0) {
            float avg_loss = epoch_loss / (float)num_samples;
            int current_percent = ((epoch + 1) * 100) / epochs;
            int is_final_epoch = (epoch + 1) == epochs;

            if (current_percent >= next_progress || is_final_epoch) {
                printf("[train] %d%% epoch=%d/%d loss=%.6f\n",
                       current_percent,
                       epoch + 1,
                       epochs,
                       avg_loss);

                while (next_progress <= current_percent && next_progress > 0) {
                    next_progress += progress_percent;
                }
            }
        }
    }

    if (final_loss_out) {
        *final_loss_out = epoch_loss / (float)num_samples;
    }

    status = 0;

cleanup:
    free_grad_accumulators(acc_w, acc_b, model->num_layers);
    free(grad_b_sizes);
    free(grad_w_sizes);
    free(layer_biases);
    free(layer_weights);
    free(output);
    return status;
}

int sequential_model_train(SequentialModel *model,
                           const float *inputs,
                           const float *targets,
                           int num_samples,
                           int input_size,
                           int target_size,
                           int epochs,
                           int batch_size,
                           float *final_loss_out) {
    return sequential_model_train_with_progress(model,
                                                inputs,
                                                targets,
                                                num_samples,
                                                input_size,
                                                target_size,
                                                epochs,
                                                batch_size,
                                                0,
                                                final_loss_out);
}

void sequential_train_config_init_sgd(SequentialTrainConfig *cfg,
                                      LossFunctionType loss_function,
                                      float learning_rate) {
    if (!cfg) return;
    cfg->loss_function = loss_function;
    cfg->optimizer = OPTIMIZER_SGD;
    cfg->learning_rate = learning_rate;
    cfg->optimizer_state = NULL;
    cfg->adam_state = NULL;
}

void sequential_train_config_init_optimizer(SequentialTrainConfig *cfg,
                                            LossFunctionType loss_function,
                                            OptimizerType optimizer,
                                            float learning_rate,
                                            OptimizerState *optimizer_state) {
    if (!cfg) return;
    cfg->loss_function = loss_function;
    cfg->optimizer = optimizer;
    cfg->learning_rate = learning_rate;
    cfg->optimizer_state = optimizer_state;
    cfg->adam_state = optimizer_state;
}

void sequential_train_config_init_rmsprop(SequentialTrainConfig *cfg,
                                          LossFunctionType loss_function,
                                          float learning_rate,
                                          OptimizerState *optimizer_state) {
    sequential_train_config_init_optimizer(cfg,
                                           loss_function,
                                           OPTIMIZER_RMSPROP,
                                           learning_rate,
                                           optimizer_state);
}

void sequential_train_config_init_adam(SequentialTrainConfig *cfg,
                                       LossFunctionType loss_function,
                                       float learning_rate,
                                       AdamOptimizerState *adam_state) {
    sequential_train_config_init_optimizer(cfg,
                                           loss_function,
                                           OPTIMIZER_ADAM,
                                           learning_rate,
                                           adam_state);
}

void sequential_model_optimizer_state_free(SequentialModel *model,
                                           OptimizerState *state) {
    if (!model || !state) return;

    if (state->m_w) {
        for (int i = 0; i < model->num_layers; i++) {
            free(state->m_w[i]);
        }
        free(state->m_w);
    }

    if (state->v_w) {
        for (int i = 0; i < model->num_layers; i++) {
            free(state->v_w[i]);
        }
        free(state->v_w);
    }

    if (state->m_b) {
        for (int i = 0; i < model->num_layers; i++) {
            free(state->m_b[i]);
        }
        free(state->m_b);
    }

    if (state->v_b) {
        for (int i = 0; i < model->num_layers; i++) {
            free(state->v_b[i]);
        }
        free(state->v_b);
    }

    state->m_w = NULL;
    state->v_w = NULL;
    state->m_b = NULL;
    state->v_b = NULL;
    state->step = 0;
    state->beta1 = 0.0f;
    state->beta2 = 0.0f;
}

int sequential_model_optimizer_state_init(SequentialModel *model,
                                          OptimizerState *out_state,
                                          OptimizerType optimizer,
                                          float beta1,
                                          float beta2) {
    if (!model || model->num_layers <= 0 || !out_state) return -1;
    if (optimizer != OPTIMIZER_SGD && optimizer != OPTIMIZER_ADAM && optimizer != OPTIMIZER_RMSPROP && optimizer != OPTIMIZER_ADAGRAD) return -1;
    if (optimizer == OPTIMIZER_SGD) return 0;
    if (beta1 <= 0.0f || beta1 >= 1.0f) return -1;
    if (optimizer == OPTIMIZER_ADAM && (beta2 <= 0.0f || beta2 >= 1.0f)) return -1;

    if (out_state->m_w || out_state->v_w || out_state->m_b || out_state->v_b) return -1;

    out_state->m_w = calloc((size_t)model->num_layers, sizeof(float *));
    out_state->v_w = calloc((size_t)model->num_layers, sizeof(float *));
    out_state->m_b = calloc((size_t)model->num_layers, sizeof(float *));
    out_state->v_b = calloc((size_t)model->num_layers, sizeof(float *));
    if (!out_state->m_w || !out_state->v_w || !out_state->m_b || !out_state->v_b) {
        sequential_model_optimizer_state_free(model, out_state);
        return -1;
    }

    for (int i = 0; i < model->num_layers; i++) {
        int w_size = model->layers[i].weights_size(model->layers[i].ctx);
        int b_size = model->layers[i].biases_size(model->layers[i].ctx);
        out_state->m_w[i] = calloc((size_t)w_size, sizeof(float));
        out_state->v_w[i] = (optimizer == OPTIMIZER_ADAM) ? calloc((size_t)w_size, sizeof(float)) : NULL;
        out_state->m_b[i] = calloc((size_t)b_size, sizeof(float));
        out_state->v_b[i] = (optimizer == OPTIMIZER_ADAM) ? calloc((size_t)b_size, sizeof(float)) : NULL;
        if (!out_state->m_w[i] || !out_state->m_b[i] ||
            (optimizer == OPTIMIZER_ADAM && (!out_state->v_w[i] || !out_state->v_b[i]))) {
            sequential_model_optimizer_state_free(model, out_state);
            return -1;
        }
    }

    out_state->step = 1;
    out_state->beta1 = beta1;
    out_state->beta2 = (optimizer == OPTIMIZER_ADAM) ? beta2 : 0.0f;
    return 0;
}

void sequential_model_adam_state_free(SequentialModel *model,
                                      AdamOptimizerState *state) {
    sequential_model_optimizer_state_free(model, state);
}

int sequential_model_adam_state_init(SequentialModel *model,
                                     AdamOptimizerState *out_state,
                                     float beta1,
                                     float beta2) {
    return sequential_model_optimizer_state_init(model,
                                                 out_state,
                                                 OPTIMIZER_ADAM,
                                                 beta1,
                                                 beta2);
}

int sequential_model_train_step_cfg(SequentialModel *model,
                                    const float *input,
                                    const float *target,
                                    float *output,
                                    const SequentialTrainConfig *cfg,
                                    float *loss_out) {
    if (!cfg) return -1;
    return sequential_model_train_step(model,
                                       input,
                                       target,
                                       output,
                                       cfg->loss_function,
                                       cfg->optimizer,
                                       cfg->learning_rate,
                                       cfg->optimizer_state ? cfg->optimizer_state : cfg->adam_state,
                                       loss_out);
}

int sequential_model_train_step(SequentialModel *model,
                                const float *input,
                                const float *target,
                                float *output,
                                LossFunctionType loss_function,
                                OptimizerType optimizer,
                                float learning_rate,
                                OptimizerState *optimizer_state,
                                float *loss_out) {
    if (!model || model->num_layers <= 0 || !input || !target || !output) {
        return -1;
    }

    if (sequential_model_forward(model, input, output) != 0) {
        return -1;
    }

    return sequential_model_optimize_from_prediction(model,
                                                     output,
                                                     target,
                                                     loss_function,
                                                     optimizer,
                                                     learning_rate,
                                                     optimizer_state,
                                                     loss_out);
}

int sequential_model_optimize_from_prediction(SequentialModel *model,
                                              const float *prediction,
                                              const float *target,
                                              LossFunctionType loss_function,
                                              OptimizerType optimizer,
                                              float learning_rate,
                                              OptimizerState *optimizer_state,
                                              float *loss_out) {
    if (!model || model->num_layers <= 0 || !prediction || !target) {
        return -1;
    }

    int output_size = model->layers[model->num_layers - 1].output_size(
        model->layers[model->num_layers - 1].ctx);

    if (!optimizer_state_valid(optimizer, optimizer_state)) {
        return -1;
    }

    if (!model->work_delta_a || !model->work_delta_b ||
        !model->work_grad_w  || !model->work_grad_b) {
        int width = 0, max_grad_w_size = 0, max_grad_b_size = 0;
        for (int i = 0; i < model->num_layers; i++) {
            int in  = model->layers[i].input_size(model->layers[i].ctx);
            int out = model->layers[i].output_size(model->layers[i].ctx);
            int ws  = model->layers[i].weights_size(model->layers[i].ctx);
            int bs  = model->layers[i].biases_size(model->layers[i].ctx);
            if (in  > width)            width            = in;
            if (out > width)            width            = out;
            if (ws  > max_grad_w_size)  max_grad_w_size  = ws;
            if (bs  > max_grad_b_size)  max_grad_b_size  = bs;
        }
        if (max_grad_w_size <= 0 || max_grad_b_size <= 0) return -1;
        if (ensure_workspace(&model->work_delta_a, &model->work_delta_size, width) != 0 ||
            ensure_workspace(&model->work_delta_b, &model->work_delta_size, width) != 0 ||
            ensure_workspace(&model->work_grad_w,  &model->work_grad_w_size, max_grad_w_size) != 0 ||
            ensure_workspace(&model->work_grad_b,  &model->work_grad_b_size, max_grad_b_size) != 0) {
            return -1;
        }
    }

    float *delta_curr = model->work_delta_a;
    float *delta_prev = model->work_delta_b;
    float *grad_w = model->work_grad_w;
    float *grad_b = model->work_grad_b;

    if (compute_loss_and_grad(loss_function, prediction, target, output_size, loss_out, delta_curr) != 0) {
        return -1;
    }

    for (int i = model->num_layers - 1; i >= 0; i--) {
        float *next_delta = (i > 0) ? delta_prev : NULL;

        int grad_w_size = model->layers[i].weights_size(model->layers[i].ctx);
        int grad_b_size = model->layers[i].biases_size(model->layers[i].ctx);

        if (model->layers[i].backward(model->layers[i].ctx, delta_curr, next_delta, grad_w, grad_b) != 0) {
            return -1;
        }

        float *opt_state_w_a = NULL;
        float *opt_state_w_b = NULL;
        float *opt_state_b_a = NULL;
        float *opt_state_b_b = NULL;

        if (optimizer == OPTIMIZER_ADAM) {
            if (!optimizer_state->m_w[i] || !optimizer_state->v_w[i] ||
                !optimizer_state->m_b[i] || !optimizer_state->v_b[i]) {
                return -1;
            }

            opt_state_w_a = optimizer_state->m_w[i];
            opt_state_w_b = optimizer_state->v_w[i];
            opt_state_b_a = optimizer_state->m_b[i];
            opt_state_b_b = optimizer_state->v_b[i];
        } else if (optimizer == OPTIMIZER_RMSPROP) {
            if (!optimizer_state->m_w[i] || !optimizer_state->m_b[i]) {
                return -1;
            }

            opt_state_w_a = optimizer_state->m_w[i];
            opt_state_b_a = optimizer_state->m_b[i];
        } else if (optimizer == OPTIMIZER_ADAGRAD) {
            if (!optimizer_state->m_w[i] || !optimizer_state->m_b[i]) {
                return -1;
            }

            opt_state_w_a = optimizer_state->m_w[i];
            opt_state_b_a = optimizer_state->m_b[i];
        }

        if (apply_optimizer_update(model->layers[i].weights(model->layers[i].ctx),
                                   grad_w,
                                   grad_w_size,
                                   optimizer,
                                   learning_rate,
                                   optimizer_state,
                                   opt_state_w_a,
                                   opt_state_w_b) != 0 ||
            apply_optimizer_update(model->layers[i].biases(model->layers[i].ctx),
                                   grad_b,
                                   grad_b_size,
                                   optimizer,
                                   learning_rate,
                                   optimizer_state,
                                   opt_state_b_a,
                                   opt_state_b_b) != 0) {
            return -1;
        }

        if (i > 0) {
            float *tmp = delta_curr;
            delta_curr = delta_prev;
            delta_prev = tmp;
        }
    }

    if (optimizer == OPTIMIZER_ADAM) {
        optimizer_state->step += 1;
    }

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

int sequential_train_step(Layer *layers, int num_layers,
                          const float *input, const float *target,
                          float *output,
                          float **grads_w, float **grads_b,
                          LossFunctionType loss_function,
                          OptimizerType optimizer,
                          float learning_rate,
                          OptimizerState *optimizer_state,
                          float *loss_out) {
    if (!layers || num_layers <= 0 || !input || !target || !output || !grads_w || !grads_b) {
        return -1;
    }

    if (sequential_forward(layers, num_layers, input, output) != 0) {
        return -1;
    }

    return sequential_optimize_from_prediction(layers,
                                               num_layers,
                                               output,
                                               target,
                                               grads_w,
                                               grads_b,
                                               loss_function,
                                               optimizer,
                                               learning_rate,
                                               optimizer_state,
                                               loss_out);
}

int sequential_optimize_from_prediction(Layer *layers, int num_layers,
                                        const float *prediction, const float *target,
                                        float **grads_w, float **grads_b,
                                        LossFunctionType loss_function,
                                        OptimizerType optimizer,
                                        float learning_rate,
                                        OptimizerState *optimizer_state,
                                        float *loss_out) {
    if (!layers || num_layers <= 0 || !prediction || !target || !grads_w || !grads_b) {
        return -1;
    }

    if (!optimizer_state_valid(optimizer, optimizer_state)) {
        return -1;
    }

    int output_size = layers[num_layers - 1].output_size;

    int width = max_layer_width(layers, num_layers);
    float *delta_curr = malloc((size_t)width * sizeof(float));
    float *delta_prev = malloc((size_t)width * sizeof(float));
    if (!delta_curr || !delta_prev) {
        free(delta_curr);
        free(delta_prev);
        return -1;
    }

    if (compute_loss_and_grad(loss_function, prediction, target, output_size, loss_out, delta_curr) != 0) {
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

        float *opt_state_w_a = NULL;
        float *opt_state_w_b = NULL;
        float *opt_state_b_a = NULL;
        float *opt_state_b_b = NULL;

        if (optimizer == OPTIMIZER_ADAM) {
            if (!optimizer_state->m_w[i] || !optimizer_state->v_w[i] ||
                !optimizer_state->m_b[i] || !optimizer_state->v_b[i]) {
                free(delta_curr);
                free(delta_prev);
                return -1;
            }

            opt_state_w_a = optimizer_state->m_w[i];
            opt_state_w_b = optimizer_state->v_w[i];
            opt_state_b_a = optimizer_state->m_b[i];
            opt_state_b_b = optimizer_state->v_b[i];
        } else if (optimizer == OPTIMIZER_RMSPROP) {
            if (!optimizer_state->m_w[i] || !optimizer_state->m_b[i]) {
                free(delta_curr);
                free(delta_prev);
                return -1;
            }

            opt_state_w_a = optimizer_state->m_w[i];
            opt_state_b_a = optimizer_state->m_b[i];
        } else if (optimizer == OPTIMIZER_ADAGRAD) {
            if (!optimizer_state->m_w[i] || !optimizer_state->m_b[i]) {
                free(delta_curr);
                free(delta_prev);
                return -1;
            }

            opt_state_w_a = optimizer_state->m_w[i];
            opt_state_b_a = optimizer_state->m_b[i];
        }

        if (apply_optimizer_update(layers[i].weights,
                                   grads_w[i],
                                   grad_w_size,
                                   optimizer,
                                   learning_rate,
                                   optimizer_state,
                                   opt_state_w_a,
                                   opt_state_w_b) != 0 ||
            apply_optimizer_update(layers[i].biases,
                                   grads_b[i],
                                   grad_b_size,
                                   optimizer,
                                   learning_rate,
                                   optimizer_state,
                                   opt_state_b_a,
                                   opt_state_b_b) != 0) {
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

    if (optimizer == OPTIMIZER_ADAM) {
        optimizer_state->step += 1;
    }

    return 0;
}
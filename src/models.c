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

static int adam_state_valid(const AdamOptimizerState *adam_state) {
    if (!adam_state) return 0;
    if (!adam_state->m_w || !adam_state->v_w || !adam_state->m_b || !adam_state->v_b) return 0;
    if (adam_state->step <= 0) return 0;
    if (adam_state->beta1 <= 0.0f || adam_state->beta1 >= 1.0f) return 0;
    if (adam_state->beta2 <= 0.0f || adam_state->beta2 >= 1.0f) return 0;
    return 1;
}

static int apply_optimizer_update(float *weights,
                                  const float *grads,
                                  int size,
                                  OptimizerType optimizer,
                                  float learning_rate,
                                  AdamOptimizerState *adam_state,
                                  float *adam_m,
                                  float *adam_v) {
    if (!weights || !grads || size <= 0) return -1;

    if (optimizer == OPTIMIZER_SGD) {
        return sgd_optimizer(weights, (float *)grads, learning_rate, size);
    }

    if (optimizer == OPTIMIZER_ADAM) {
        if (!adam_state || !adam_m || !adam_v) return -1;
        return adam_optimizer(weights,
                              (float *)grads,
                              adam_m,
                              adam_v,
                              adam_state->beta1,
                              adam_state->beta2,
                              learning_rate,
                              adam_state->step,
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

    return -1;
}

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
    model->compiled_owns_adam_state = 0;
    model->compiled_adam_state = (AdamOptimizerState){0};
    return 0;
}

void sequential_model_free(SequentialModel *model) {
    if (!model) return;

    if (model->compiled_owns_adam_state) {
        sequential_model_adam_state_free(model, &model->compiled_adam_state);
        model->compiled_owns_adam_state = 0;
    }

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

    if (model->compiled_owns_adam_state) {
        sequential_model_adam_state_free(model, &model->compiled_adam_state);
        model->compiled_owns_adam_state = 0;
    }
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

int sequential_model_compile(SequentialModel *model,
                             LossFunctionType loss_function,
                             OptimizerType optimizer,
                             float learning_rate,
                             float adam_beta1,
                             float adam_beta2) {
    if (!model || model->num_layers <= 0 || learning_rate <= 0.0f) return -1;
    if (loss_function != LOSS_MSE && loss_function != LOSS_BCE) return -1;
    if (optimizer != OPTIMIZER_SGD && optimizer != OPTIMIZER_ADAM) return -1;

    if (model->compiled_owns_adam_state) {
        sequential_model_adam_state_free(model, &model->compiled_adam_state);
        model->compiled_owns_adam_state = 0;
    }

    model->compiled = 0;
    model->compiled_loss = loss_function;
    model->compiled_optimizer = optimizer;
    model->compiled_learning_rate = learning_rate;

    if (optimizer == OPTIMIZER_ADAM) {
        if (sequential_model_adam_state_init(model,
                                             &model->compiled_adam_state,
                                             adam_beta1,
                                             adam_beta2) != 0) {
            return -1;
        }
        model->compiled_owns_adam_state = 1;
    }

    model->compiled = 1;
    return 0;
}

int sequential_model_train(SequentialModel *model,
                           const float *inputs,
                           const float *targets,
                           int num_samples,
                           int input_size,
                           int target_size,
                           int epochs,
                           float *final_loss_out) {
    if (!model || !inputs || !targets || num_samples <= 0 || input_size <= 0 || target_size <= 0 || epochs <= 0) {
        return -1;
    }
    if (!model->compiled) return -1;
    if (model->num_layers <= 0) return -1;

    int expected_input_size = model->layers[0].input_size(model->layers[0].ctx);
    int expected_target_size = model->layers[model->num_layers - 1].output_size(model->layers[model->num_layers - 1].ctx);
    if (input_size != expected_input_size || target_size != expected_target_size) return -1;

    float *output = malloc((size_t)expected_target_size * sizeof(float));
    if (!output) return -1;

    float epoch_loss = 0.0f;
    for (int epoch = 0; epoch < epochs; epoch++) {
        epoch_loss = 0.0f;
        for (int i = 0; i < num_samples; i++) {
            const float *sample_input = inputs + ((size_t)i * (size_t)input_size);
            const float *sample_target = targets + ((size_t)i * (size_t)target_size);
            float step_loss = 0.0f;
            AdamOptimizerState *adam_state = (model->compiled_optimizer == OPTIMIZER_ADAM)
                                             ? &model->compiled_adam_state
                                             : NULL;

            if (sequential_model_train_step_with_loss(model,
                                                      sample_input,
                                                      sample_target,
                                                      output,
                                                      model->compiled_loss,
                                                      model->compiled_optimizer,
                                                      model->compiled_learning_rate,
                                                      adam_state,
                                                      &step_loss) != 0) {
                free(output);
                return -1;
            }

            epoch_loss += step_loss;
        }
    }

    if (final_loss_out) {
        *final_loss_out = epoch_loss / (float)num_samples;
    }

    free(output);
    return 0;
}

int sequential_model_fit(SequentialModel *model,
                         const float *inputs,
                         const float *targets,
                         int num_samples,
                         int input_size,
                         int target_size,
                         int epochs,
                         float *final_loss_out) {
    return sequential_model_train(model,
                                  inputs,
                                  targets,
                                  num_samples,
                                  input_size,
                                  target_size,
                                  epochs,
                                  final_loss_out);
}

void sequential_train_config_init_sgd(SequentialTrainConfig *cfg,
                                      LossFunctionType loss_function,
                                      float learning_rate) {
    if (!cfg) return;
    cfg->loss_function = loss_function;
    cfg->optimizer = OPTIMIZER_SGD;
    cfg->learning_rate = learning_rate;
    cfg->adam_state = NULL;
}

void sequential_train_config_init_adam(SequentialTrainConfig *cfg,
                                       LossFunctionType loss_function,
                                       float learning_rate,
                                       AdamOptimizerState *adam_state) {
    if (!cfg) return;
    cfg->loss_function = loss_function;
    cfg->optimizer = OPTIMIZER_ADAM;
    cfg->learning_rate = learning_rate;
    cfg->adam_state = adam_state;
}

void sequential_model_adam_state_free(SequentialModel *model,
                                      AdamOptimizerState *state) {
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

int sequential_model_adam_state_init(SequentialModel *model,
                                     AdamOptimizerState *out_state,
                                     float beta1,
                                     float beta2) {
    if (!model || model->num_layers <= 0 || !out_state) return -1;
    if (beta1 <= 0.0f || beta1 >= 1.0f || beta2 <= 0.0f || beta2 >= 1.0f) return -1;
    if (out_state->m_w || out_state->v_w || out_state->m_b || out_state->v_b) return -1;

    out_state->m_w = calloc((size_t)model->num_layers, sizeof(float *));
    out_state->v_w = calloc((size_t)model->num_layers, sizeof(float *));
    out_state->m_b = calloc((size_t)model->num_layers, sizeof(float *));
    out_state->v_b = calloc((size_t)model->num_layers, sizeof(float *));
    if (!out_state->m_w || !out_state->v_w || !out_state->m_b || !out_state->v_b) {
        sequential_model_adam_state_free(model, out_state);
        return -1;
    }

    for (int i = 0; i < model->num_layers; i++) {
        int w_size = model->layers[i].weights_size(model->layers[i].ctx);
        int b_size = model->layers[i].biases_size(model->layers[i].ctx);
        out_state->m_w[i] = calloc((size_t)w_size, sizeof(float));
        out_state->v_w[i] = calloc((size_t)w_size, sizeof(float));
        out_state->m_b[i] = calloc((size_t)b_size, sizeof(float));
        out_state->v_b[i] = calloc((size_t)b_size, sizeof(float));
        if (!out_state->m_w[i] || !out_state->v_w[i] || !out_state->m_b[i] || !out_state->v_b[i]) {
            sequential_model_adam_state_free(model, out_state);
            return -1;
        }
    }

    out_state->step = 1;
    out_state->beta1 = beta1;
    out_state->beta2 = beta2;
    return 0;
}

int sequential_model_train_step_cfg(SequentialModel *model,
                                    const float *input,
                                    const float *target,
                                    float *output,
                                    const SequentialTrainConfig *cfg,
                                    float *loss_out) {
    if (!cfg) return -1;
    return sequential_model_train_step_with_loss(model,
                                                 input,
                                                 target,
                                                 output,
                                                 cfg->loss_function,
                                                 cfg->optimizer,
                                                 cfg->learning_rate,
                                                 cfg->adam_state,
                                                 loss_out);
}

int sequential_model_train_step(SequentialModel *model,
                                const float *input,
                                const float *target,
                                float *output,
                                OptimizerType optimizer,
                                float learning_rate,
                                AdamOptimizerState *adam_state,
                                float *loss_out) {
    return sequential_model_train_step_with_loss(model,
                                                 input,
                                                 target,
                                                 output,
                                                 LOSS_MSE,
                                                 optimizer,
                                                 learning_rate,
                                                 adam_state,
                                                 loss_out);
}

int sequential_model_train_step_with_loss(SequentialModel *model,
                                          const float *input,
                                          const float *target,
                                          float *output,
                                          LossFunctionType loss_function,
                                          OptimizerType optimizer,
                                          float learning_rate,
                                          AdamOptimizerState *adam_state,
                                          float *loss_out) {
    if (!model || model->num_layers <= 0 || !input || !target || !output) {
        return -1;
    }

    if (sequential_model_forward(model, input, output) != 0) {
        return -1;
    }

    return sequential_model_optimize_from_prediction_with_loss(model,
                                                               output,
                                                               target,
                                                               loss_function,
                                                               optimizer,
                                                               learning_rate,
                                                               adam_state,
                                                               loss_out);
}

int sequential_model_train_step_mse(SequentialModel *model,
                                    const float *input,
                                    const float *target,
                                    float *output,
                                    OptimizerType optimizer,
                                    float learning_rate,
                                    AdamOptimizerState *adam_state,
                                    float *loss_out) {
    return sequential_model_train_step_with_loss(model,
                                                 input,
                                                 target,
                                                 output,
                                                 LOSS_MSE,
                                                 optimizer,
                                                 learning_rate,
                                                 adam_state,
                                                 loss_out);
}

int sequential_model_train_step_bce(SequentialModel *model,
                                    const float *input,
                                    const float *target,
                                    float *output,
                                    OptimizerType optimizer,
                                    float learning_rate,
                                    AdamOptimizerState *adam_state,
                                    float *loss_out) {
    return sequential_model_train_step_with_loss(model,
                                                 input,
                                                 target,
                                                 output,
                                                 LOSS_BCE,
                                                 optimizer,
                                                 learning_rate,
                                                 adam_state,
                                                 loss_out);
}

int sequential_model_optimize_from_prediction(SequentialModel *model,
                                              const float *prediction,
                                              const float *target,
                                              OptimizerType optimizer,
                                              float learning_rate,
                                              AdamOptimizerState *adam_state,
                                              float *loss_out) {
    return sequential_model_optimize_from_prediction_with_loss(model,
                                                               prediction,
                                                               target,
                                                               LOSS_MSE,
                                                               optimizer,
                                                               learning_rate,
                                                               adam_state,
                                                               loss_out);
}

int sequential_model_optimize_from_prediction_with_loss(SequentialModel *model,
                                                        const float *prediction,
                                                        const float *target,
                                                        LossFunctionType loss_function,
                                              OptimizerType optimizer,
                                              float learning_rate,
                                              AdamOptimizerState *adam_state,
                                              float *loss_out) {
    if (!model || model->num_layers <= 0 || !prediction || !target) {
        return -1;
    }

    int output_size = model->layers[model->num_layers - 1].output_size(
        model->layers[model->num_layers - 1].ctx);

    if (optimizer == OPTIMIZER_ADAM && !adam_state_valid(adam_state)) {
        return -1;
    }

    int width = max_plugin_layer_width(model);
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

        float *adam_m_w = NULL;
        float *adam_v_w = NULL;
        float *adam_m_b = NULL;
        float *adam_v_b = NULL;

        if (optimizer == OPTIMIZER_ADAM) {
            if (!adam_state->m_w[i] || !adam_state->v_w[i] || !adam_state->m_b[i] || !adam_state->v_b[i]) {
                free(grad_w);
                free(grad_b);
                free(delta_curr);
                free(delta_prev);
                return -1;
            }

            adam_m_w = adam_state->m_w[i];
            adam_v_w = adam_state->v_w[i];
            adam_m_b = adam_state->m_b[i];
            adam_v_b = adam_state->v_b[i];
        }

        if (apply_optimizer_update(model->layers[i].weights(model->layers[i].ctx),
                                   grad_w,
                                   grad_w_size,
                                   optimizer,
                                   learning_rate,
                                   adam_state,
                                   adam_m_w,
                                   adam_v_w) != 0 ||
            apply_optimizer_update(model->layers[i].biases(model->layers[i].ctx),
                                   grad_b,
                                   grad_b_size,
                                   optimizer,
                                   learning_rate,
                                   adam_state,
                                   adam_m_b,
                                   adam_v_b) != 0) {
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

    if (optimizer == OPTIMIZER_ADAM) {
        adam_state->step += 1;
    }

    return 0;
}

int sequential_model_optimize_from_prediction_mse(SequentialModel *model,
                                                  const float *prediction,
                                                  const float *target,
                                                  OptimizerType optimizer,
                                                  float learning_rate,
                                                  AdamOptimizerState *adam_state,
                                                  float *loss_out) {
    return sequential_model_optimize_from_prediction_with_loss(model,
                                                               prediction,
                                                               target,
                                                               LOSS_MSE,
                                                               optimizer,
                                                               learning_rate,
                                                               adam_state,
                                                               loss_out);
}

int sequential_model_optimize_from_prediction_bce(SequentialModel *model,
                                                  const float *prediction,
                                                  const float *target,
                                                  OptimizerType optimizer,
                                                  float learning_rate,
                                                  AdamOptimizerState *adam_state,
                                                  float *loss_out) {
    return sequential_model_optimize_from_prediction_with_loss(model,
                                                               prediction,
                                                               target,
                                                               LOSS_BCE,
                                                               optimizer,
                                                               learning_rate,
                                                               adam_state,
                                                               loss_out);
}

int sequential_model_train_step_sgd(SequentialModel *model,
                                    const float *input,
                                    const float *target,
                                    float *output,
                                    float learning_rate,
                                    float *loss_out) {
    return sequential_model_train_step(model,
                                       input,
                                       target,
                                       output,
                                       OPTIMIZER_SGD,
                                       learning_rate,
                                       NULL,
                                       loss_out);
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
                          OptimizerType optimizer,
                          float learning_rate,
                          AdamOptimizerState *adam_state,
                          float *loss_out) {
    return sequential_train_step_with_loss(layers,
                                           num_layers,
                                           input,
                                           target,
                                           output,
                                           grads_w,
                                           grads_b,
                                           LOSS_MSE,
                                           optimizer,
                                           learning_rate,
                                           adam_state,
                                           loss_out);
}

int sequential_train_step_with_loss(Layer *layers, int num_layers,
                                    const float *input, const float *target,
                                    float *output,
                                    float **grads_w, float **grads_b,
                                    LossFunctionType loss_function,
                                    OptimizerType optimizer,
                                    float learning_rate,
                                    AdamOptimizerState *adam_state,
                                    float *loss_out) {
    if (!layers || num_layers <= 0 || !input || !target || !output || !grads_w || !grads_b) {
        return -1;
    }

    if (sequential_forward(layers, num_layers, input, output) != 0) {
        return -1;
    }

    return sequential_optimize_from_prediction_with_loss(layers,
                                                         num_layers,
                                                         output,
                                                         target,
                                                         grads_w,
                                                         grads_b,
                                                         loss_function,
                                                         optimizer,
                                                         learning_rate,
                                                         adam_state,
                                                         loss_out);
}

int sequential_train_step_mse(Layer *layers, int num_layers,
                              const float *input, const float *target,
                              float *output,
                              float **grads_w, float **grads_b,
                              OptimizerType optimizer,
                              float learning_rate,
                              AdamOptimizerState *adam_state,
                              float *loss_out) {
    return sequential_train_step_with_loss(layers,
                                           num_layers,
                                           input,
                                           target,
                                           output,
                                           grads_w,
                                           grads_b,
                                           LOSS_MSE,
                                           optimizer,
                                           learning_rate,
                                           adam_state,
                                           loss_out);
}

int sequential_train_step_bce(Layer *layers, int num_layers,
                              const float *input, const float *target,
                              float *output,
                              float **grads_w, float **grads_b,
                              OptimizerType optimizer,
                              float learning_rate,
                              AdamOptimizerState *adam_state,
                              float *loss_out) {
    return sequential_train_step_with_loss(layers,
                                           num_layers,
                                           input,
                                           target,
                                           output,
                                           grads_w,
                                           grads_b,
                                           LOSS_BCE,
                                           optimizer,
                                           learning_rate,
                                           adam_state,
                                           loss_out);
}

int sequential_optimize_from_prediction(Layer *layers, int num_layers,
                                        const float *prediction, const float *target,
                                        float **grads_w, float **grads_b,
                                        OptimizerType optimizer,
                                        float learning_rate,
                                        AdamOptimizerState *adam_state,
                                        float *loss_out) {
    return sequential_optimize_from_prediction_with_loss(layers,
                                                         num_layers,
                                                         prediction,
                                                         target,
                                                         grads_w,
                                                         grads_b,
                                                         LOSS_MSE,
                                                         optimizer,
                                                         learning_rate,
                                                         adam_state,
                                                         loss_out);
}

int sequential_optimize_from_prediction_with_loss(Layer *layers, int num_layers,
                                                  const float *prediction, const float *target,
                                                  float **grads_w, float **grads_b,
                                                  LossFunctionType loss_function,
                                        OptimizerType optimizer,
                                        float learning_rate,
                                        AdamOptimizerState *adam_state,
                                        float *loss_out) {
    if (!layers || num_layers <= 0 || !prediction || !target || !grads_w || !grads_b) {
        return -1;
    }

    if (optimizer == OPTIMIZER_ADAM && !adam_state_valid(adam_state)) {
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

        float *adam_m_w = NULL;
        float *adam_v_w = NULL;
        float *adam_m_b = NULL;
        float *adam_v_b = NULL;

        if (optimizer == OPTIMIZER_ADAM) {
            if (!adam_state->m_w[i] || !adam_state->v_w[i] || !adam_state->m_b[i] || !adam_state->v_b[i]) {
                free(delta_curr);
                free(delta_prev);
                return -1;
            }

            adam_m_w = adam_state->m_w[i];
            adam_v_w = adam_state->v_w[i];
            adam_m_b = adam_state->m_b[i];
            adam_v_b = adam_state->v_b[i];
        }

        if (apply_optimizer_update(layers[i].weights,
                                   grads_w[i],
                                   grad_w_size,
                                   optimizer,
                                   learning_rate,
                                   adam_state,
                                   adam_m_w,
                                   adam_v_w) != 0 ||
            apply_optimizer_update(layers[i].biases,
                                   grads_b[i],
                                   grad_b_size,
                                   optimizer,
                                   learning_rate,
                                   adam_state,
                                   adam_m_b,
                                   adam_v_b) != 0) {
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
        adam_state->step += 1;
    }

    return 0;
}

int sequential_optimize_from_prediction_mse(Layer *layers, int num_layers,
                                            const float *prediction, const float *target,
                                            float **grads_w, float **grads_b,
                                            OptimizerType optimizer,
                                            float learning_rate,
                                            AdamOptimizerState *adam_state,
                                            float *loss_out) {
    return sequential_optimize_from_prediction_with_loss(layers,
                                                         num_layers,
                                                         prediction,
                                                         target,
                                                         grads_w,
                                                         grads_b,
                                                         LOSS_MSE,
                                                         optimizer,
                                                         learning_rate,
                                                         adam_state,
                                                         loss_out);
}

int sequential_optimize_from_prediction_bce(Layer *layers, int num_layers,
                                            const float *prediction, const float *target,
                                            float **grads_w, float **grads_b,
                                            OptimizerType optimizer,
                                            float learning_rate,
                                            AdamOptimizerState *adam_state,
                                            float *loss_out) {
    return sequential_optimize_from_prediction_with_loss(layers,
                                                         num_layers,
                                                         prediction,
                                                         target,
                                                         grads_w,
                                                         grads_b,
                                                         LOSS_BCE,
                                                         optimizer,
                                                         learning_rate,
                                                         adam_state,
                                                         loss_out);
}

int sequential_train_step_sgd(Layer *layers, int num_layers,
                              const float *input, const float *target,
                              float *output,
                              float **grads_w, float **grads_b,
                              float learning_rate,
                              float *loss_out) {
    return sequential_train_step(layers,
                                 num_layers,
                                 input,
                                 target,
                                 output,
                                 grads_w,
                                 grads_b,
                                 OPTIMIZER_SGD,
                                 learning_rate,
                                 NULL,
                                 loss_out);
}
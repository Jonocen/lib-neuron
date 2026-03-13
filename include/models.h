#ifndef MODELS_H
#define MODELS_H

#include "layers.h"
#include "lossfunctions.h"
#include "matrixcalculation.h"
#include "optimizers.h"

typedef struct {
	LayerPlugin *layers;
	int          num_layers;
	int          capacity;
} SequentialModel;

/*
 * Initializes a dynamic sequential model container.
 * Returns 0 on success, -1 on invalid input or allocation failure.
 */
int sequential_model_init(SequentialModel *model, int initial_capacity);

/*
 * Frees all contained layer plugins and internal storage.
 */
void sequential_model_free(SequentialModel *model);

/*
 * Adds a plugin layer to the model. Ownership of `layer` is moved into model.
 * Returns 0 on success, -1 on invalid input or allocation failure.
 */
int sequential_model_add_layer(SequentialModel *model, LayerPlugin layer);

/*
 * Convenience helper for adding a dense layer plugin.
 * Returns 0 on success, -1 on failure.
 */
int sequential_model_add_dense(SequentialModel *model,
							   int input_size,
							   int output_size,
							   Activation activation);

/*
 * Runs forward pass for all layers in a sequential model.
 * `output` must have at least output size of the last layer.
 * Returns 0 on success, -1 on invalid input or failure.
 */
int sequential_model_forward(SequentialModel *model,
							 const float *input,
							 float *output);

/*
 * One training step using MSE loss + SGD on a sequential model.
 * Uses lossfunctions and optimizers modules internally.
 * - `loss_out` is optional and may be NULL.
 * Returns 0 on success, -1 on invalid input or failure.
 */
int sequential_model_train_step_sgd(SequentialModel *model,
									const float *input,
									const float *target,
									float *output,
									float learning_rate,
									float *loss_out);

/*
 * Runs a forward pass through all layers.
 * `output` must have at least layers[num_layers - 1].output_size elements.
 * Returns 0 on success, -1 on invalid input or failure.
 */
int sequential_forward(Layer *layers, int num_layers, const float *input, float *output);

/*
 * Performs one training step with MSE loss + SGD updates.
 * - `grads_w[i]` must point to a buffer of size output_size * input_size for layer i.
 * - `grads_b[i]` must point to a buffer of size output_size for layer i.
 * - `loss_out` is optional and may be NULL.
 * Returns 0 on success, -1 on invalid input or failure.
 */
int sequential_train_step_sgd(Layer *layers, int num_layers,
							  const float *input, const float *target,
							  float *output,
							  float **grads_w, float **grads_b,
							  float learning_rate,
							  float *loss_out);

#endif /* MODELS_H */

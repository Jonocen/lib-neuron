#ifndef MODELS_H
#define MODELS_H

#include "layers.h"
#include "lossfunctions.h"
#include "matrixcalculation.h"
#include "optimizers.h"

typedef struct {
	float **m_w;
	float **v_w;
	float **m_b;
	float **v_b;
	int     step;
	float   beta1;
	float   beta2;
} AdamOptimizerState;

typedef struct {
	LayerPlugin *layers;
	int          num_layers;
	int          capacity;
	int          compiled;
	LossFunctionType compiled_loss;
	OptimizerType    compiled_optimizer;
	float            compiled_learning_rate;
	int              compiled_owns_adam_state;
	AdamOptimizerState compiled_adam_state;
	/* Internal reusable workspaces for faster forward/backward passes. */
	float           *work_forward_a;
	float           *work_forward_b;
	int              work_forward_size;
	float           *work_delta_a;
	float           *work_delta_b;
	int              work_delta_size;
	float           *work_grad_w;
	float           *work_grad_b;
	int              work_grad_w_size;
	int              work_grad_b_size;
} SequentialModel;

typedef struct {
	LossFunctionType   loss_function;
	OptimizerType      optimizer;
	float              learning_rate;
	AdamOptimizerState *adam_state;
} SequentialTrainConfig;

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
 * Convenience helper for adding a Conv2D plugin layer.
 * Input and output tensors are flattened in CHW order.
 */
int sequential_model_add_conv2d(SequentialModel *model,
								int input_width,
								int input_height,
								int input_channels,
								int output_channels,
								int kernel_width,
								int kernel_height,
								int stride,
								int padding,
								Activation activation);

/*
 * Convenience helper for adding a MaxPool2D plugin layer.
 * Input and output tensors are flattened in CHW order.
 */
int sequential_model_add_maxpool2d(SequentialModel *model,
								   int input_width,
								   int input_height,
								   int channels,
								   int pool_width,
								   int pool_height,
								   int stride,
								   int padding);

/*
 * Runs forward pass for all layers in a sequential model.
 * `output` must have at least output size of the last layer.
 * Returns 0 on success, -1 on invalid input or failure.
 */
int sequential_model_forward(SequentialModel *model,
							 const float *input,
							 float *output);

/*
 * Random-initializes all layer weights and biases in range
 * [-init_scale/2, +init_scale/2].
 * Returns 0 on success, -1 on invalid input.
 */
int sequential_model_randomize(SequentialModel *model, float init_scale);

/*
 * Alias for forward pass to match ML-framework style naming.
 */
int sequential_model_predict(SequentialModel *model,
							 const float *input,
							 float *output);

/*
 * Saves all layer weights and biases into a `.lnn` file.
 * Returns 0 on success, -1 on invalid input or I/O failure.
 */
int sequential_model_save_lnn(const SequentialModel *model,
						  const char *file_path);

/*
 * Loads all layer weights and biases from a `.lnn` file.
 * The model architecture must match the file exactly.
 * Returns 0 on success, -1 on invalid input, mismatch, or I/O failure.
 */
int sequential_model_load_lnn(SequentialModel *model,
						  const char *file_path);

/*
 * Compiles model training settings (loss, optimizer, learning rate).
 * For Adam and RMSProp, optimizer state/cache is allocated internally.
 * Returns 0 on success, -1 on invalid input or allocation failure.
 */
int sequential_model_compile(SequentialModel *model,
						 LossFunctionType loss_function,
						 OptimizerType optimizer,
						 float learning_rate,
						 float adam_beta1,
						 float adam_beta2);

/*
 * Trains model using previously compiled settings.
 * `inputs` and `targets` must be contiguous row-major buffers:
 * - inputs:  [num_samples * input_size]
 * - targets: [num_samples * target_size]
 * `batch_size` controls how many samples are grouped per gradient update:
 * - 1           : stochastic gradient descent (one update per sample)
 * - num_samples : full-batch gradient descent
 * - any value in between: mini-batch
 * Values > num_samples are clamped automatically.
 * Returns 0 on success, -1 on invalid input or failure.
 */
int sequential_model_train(SequentialModel *model,
					   const float *inputs,
					   const float *targets,
					   int num_samples,
					   int input_size,
					   int target_size,
					   int epochs,
					   int batch_size,
					   float *final_loss_out);

/*
 * Same as sequential_model_train, with optional progress output.
 * `progress_percent` controls print interval by training completion percent:
 * - <= 0: disabled
 * - 1..100: print approximately every N%% and always on final epoch
 * - > 100: treated as 100
 */
int sequential_model_train_with_progress(SequentialModel *model,
							 const float *inputs,
							 const float *targets,
							 int num_samples,
							 int input_size,
							 int target_size,
							 int epochs,
							 int batch_size,
							 int progress_percent,
							 float *final_loss_out);

/*
 * Initializes a train config for SGD and selected loss.
 */
void sequential_train_config_init_sgd(SequentialTrainConfig *cfg,
							  LossFunctionType loss_function,
							  float learning_rate);

/*
 * Initializes a train config for Adam and selected loss.
 */
void sequential_train_config_init_adam(SequentialTrainConfig *cfg,
							   LossFunctionType loss_function,
							   float learning_rate,
							   AdamOptimizerState *adam_state);

/*
 * Allocates and initializes Adam state buffers for all layers in `model`.
 * `out_state` must be an empty struct (all fields zero/NULL).
 * Returns 0 on success, -1 on invalid input or allocation failure.
 */
int sequential_model_adam_state_init(SequentialModel *model,
							 AdamOptimizerState *out_state,
							 float beta1,
							 float beta2);

/*
 * Frees Adam state buffers previously initialized with
 * `sequential_model_adam_state_init`.
 */
void sequential_model_adam_state_free(SequentialModel *model,
							  AdamOptimizerState *state);

/*
 * One-call train step driven by a compact config struct.
 * Returns 0 on success, -1 on invalid input or failure.
 */
int sequential_model_train_step_cfg(SequentialModel *model,
							const float *input,
							const float *target,
							float *output,
							const SequentialTrainConfig *cfg,
							float *loss_out);

/*
 * One training step with selectable loss and optimizer.
 * - `loss_function`: LOSS_MSE or LOSS_BCE.
 * - For OPTIMIZER_SGD: `adam_state` is ignored and may be NULL.
 * - For OPTIMIZER_ADAM: `adam_state` must be fully initialized.
 * - For OPTIMIZER_RMSPROP: `adam_state->m_w/m_b` are used as caches and
 *   `adam_state->beta1` is used as RMSProp beta.
 * - `loss_out` is optional and may be NULL.
 * Returns 0 on success, -1 on invalid input or failure.
 */
int sequential_model_train_step(SequentialModel *model,
							const float *input,
							const float *target,
							float *output,
							LossFunctionType loss_function,
							OptimizerType optimizer,
							float learning_rate,
							AdamOptimizerState *adam_state,
							float *loss_out);

#endif /* MODELS_H */

/*
 * Backward + update step for a model after an explicit forward pass.
 * `prediction` must be the latest output produced by sequential_model_forward
 * for the same model/input (backward uses cached activations in layers).
 * - `loss_function`: LOSS_MSE or LOSS_BCE.
 * Returns 0 on success, -1 on invalid input or failure.
 */
int sequential_model_optimize_from_prediction(SequentialModel *model,
									 const float *prediction,
									 const float *target,
									 LossFunctionType loss_function,
									 OptimizerType optimizer,
									 float learning_rate,
									 AdamOptimizerState *adam_state,
									 float *loss_out);

/*
 * Runs a forward pass through all layers.
 * `output` must have at least layers[num_layers - 1].output_size elements.
 * Returns 0 on success, -1 on invalid input or failure.
 */
int sequential_forward(Layer *layers, int num_layers, const float *input, float *output);

/*
 * Performs one training step with selectable loss and optimizer.
 * - `grads_w[i]` must point to output_size * input_size for layer i.
 * - `grads_b[i]` must point to output_size for layer i.
 * - `loss_function`: LOSS_MSE or LOSS_BCE.
 * - For OPTIMIZER_SGD: `adam_state` is ignored and may be NULL.
 * - For OPTIMIZER_ADAM: `adam_state` must be fully initialized.
 * - For OPTIMIZER_RMSPROP: `adam_state->m_w/m_b` are used as caches and
 *   `adam_state->beta1` is used as RMSProp beta.
 * - `loss_out` is optional and may be NULL.
 * Returns 0 on success, -1 on invalid input or failure.
 */
int sequential_train_step(Layer *layers, int num_layers,
					  const float *input, const float *target,
					  float *output,
					  float **grads_w, float **grads_b,
					  LossFunctionType loss_function,
					  OptimizerType optimizer,
					  float learning_rate,
					  AdamOptimizerState *adam_state,
					  float *loss_out);
/*
 * Backward + update step after an explicit forward pass with sequential_forward.
 * `prediction` must be the latest output produced by sequential_forward
 * for the same layers/input.
 * - `loss_function`: LOSS_MSE or LOSS_BCE.
 * Returns 0 on success, -1 on invalid input or failure.
 */
int sequential_optimize_from_prediction(Layer *layers, int num_layers,
						 const float *prediction, const float *target,
						 float **grads_w, float **grads_b,
						 LossFunctionType loss_function,
						 OptimizerType optimizer,
						 float learning_rate,
						 AdamOptimizerState *adam_state,
						 float *loss_out);
